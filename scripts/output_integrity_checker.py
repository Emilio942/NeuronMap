#!/usr/bin/env python3
"""
Output Integrity Checker for NeuronMap Interpretability Tools
============================================================

Comprehensive validation of tool outputs to ensure numerical plausibility,
detect empty matrices, NaN values, and dummy placeholder data.
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutputIntegrityChecker:
    """Comprehensive output integrity validation for NeuronMap tools."""
    
    def __init__(self):
        self.validation_rules = self._define_validation_rules()
        self.dummy_patterns = self._define_dummy_patterns()
        self.integrity_report = {}
        
    def _define_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define validation rules for each tool type."""
        
        rules = {
            'integrated_gradients': {
                'required_keys': ['attributions', 'baseline_scores'],
                'numeric_keys': ['attributions'],
                'array_keys': ['attributions'],
                'min_shape': (1, 1),
                'value_ranges': {'attributions': (-10, 10)},
                'no_nan_keys': ['attributions'],
                'no_empty_keys': ['attributions']
            },
            
            'deepshap_explainer': {
                'required_keys': ['shap_values', 'feature_importance'],
                'numeric_keys': ['shap_values', 'feature_importance'],
                'array_keys': ['shap_values'],
                'min_shape': (1, 1),
                'value_ranges': {'shap_values': (-5, 5)},
                'no_nan_keys': ['shap_values'],
                'no_empty_keys': ['shap_values']
            },
            
            'semantic_labeling': {
                'required_keys': ['semantic_labels'],
                'string_keys': ['semantic_labels'],
                'no_empty_keys': ['semantic_labels'],
                'min_string_length': 3
            },
            
            'ace_concepts': {
                'required_keys': ['extracted_concepts', 'concept_scores'],
                'numeric_keys': ['concept_scores'],  
                'array_keys': ['concept_scores'],
                'min_shape': (1,),
                'value_ranges': {'concept_scores': (0, 1)},
                'no_nan_keys': ['concept_scores'],
                'no_empty_keys': ['extracted_concepts']
            },
            
            'neuron_coverage': {
                'required_keys': ['coverage_statistics', 'layer_coverage'],
                'numeric_keys': ['coverage_statistics'],
                'value_ranges': {'coverage_statistics': (0, 1)},
                'no_nan_keys': ['coverage_statistics'],
                'no_empty_keys': ['layer_coverage']
            },
            
            'surprise_coverage': {
                'required_keys': ['surprise_statistics', 'outlier_indices'],
                'numeric_keys': ['surprise_statistics'],
                'array_keys': ['outlier_indices'],
                'value_ranges': {'surprise_statistics': (0, 100)},
                'no_nan_keys': ['surprise_statistics']
            },
            
            'wasserstein_distance': {
                'required_keys': ['wasserstein_distance'],
                'numeric_keys': ['wasserstein_distance'],
                'value_ranges': {'wasserstein_distance': (0, 1000)},
                'no_nan_keys': ['wasserstein_distance'],
                'positive_keys': ['wasserstein_distance']
            },
            
            'emd_heatmap': {
                'required_keys': ['emd_distance'],
                'numeric_keys': ['emd_distance'],
                'value_ranges': {'emd_distance': (0, 100)},
                'no_nan_keys': ['emd_distance'],
                'positive_keys': ['emd_distance']
            },
            
            'transformerlens_adapter': {
                'required_keys': ['neuron_activations', 'model_name'],
                'string_keys': ['model_name'],
                'no_empty_keys': ['neuron_activations', 'model_name'],
                'min_string_length': 3
            },
            
            'residual_stream_comparator': {
                'required_keys': ['comparison_summary', 'similarity_scores'],
                'numeric_keys': ['similarity_scores'],
                'value_ranges': {'similarity_scores': (-1, 1)},
                'no_nan_keys': ['similarity_scores'],
                'no_empty_keys': ['comparison_summary']
            },
            
            'tcav_plus_comparator': {
                'required_keys': ['similarity_metrics', 'compatibility_score'],
                'numeric_keys': ['similarity_metrics', 'compatibility_score'],
                'value_ranges': {'compatibility_score': (0, 1)},
                'no_nan_keys': ['similarity_metrics', 'compatibility_score'],
                'no_empty_keys': ['similarity_metrics']
            }
        }
        
        return rules
    
    def _define_dummy_patterns(self) -> List[str]:
        """Define patterns that indicate dummy/placeholder data."""
        
        return [
            r'dummy',
            r'placeholder',
            r'todo',
            r'fixme',
            r'test_.*_test',
            r'fake_data',
            r'mock_.*',
            r'example_.*',
            r'sample_data',
            r'not_implemented',
            r'coming_soon',
            r'tbd',
            r'null',
            r'none',
            r'empty',
            r'default_value'
        ]
    
    def validate_output(self, output_data: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
        """Validate integrity of tool output."""
        
        logger.info(f"🔍 Validating output integrity for: {tool_id}")
        
        validation_result = {
            'tool_id': tool_id,
            'overall_valid': True,
            'checks_passed': 0,
            'checks_failed': 0,
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        # Get validation rules for this tool
        rules = self.validation_rules.get(tool_id, {})
        
        if not rules:
            validation_result['warnings'].append(f"No validation rules defined for {tool_id}")
            logger.warning(f"⚠️ No validation rules for {tool_id}")
            return validation_result
        
        # Run validation checks
        checks = [
            self._check_required_keys,
            self._check_numeric_validity,
            self._check_array_properties,
            self._check_value_ranges,
            self._check_nan_values,
            self._check_empty_data,
            self._check_string_validity,
            self._check_dummy_patterns,
            self._check_positive_values,
            self._check_data_consistency
        ]
        
        for check_func in checks:
            try:
                check_result = check_func(output_data, rules, tool_id)
                
                if check_result['passed']:
                    validation_result['checks_passed'] += 1
                else:
                    validation_result['checks_failed'] += 1
                    validation_result['overall_valid'] = False
                    validation_result['errors'].extend(check_result.get('errors', []))
                
                validation_result['warnings'].extend(check_result.get('warnings', []))
                validation_result['details'][check_func.__name__] = check_result
                
            except Exception as e:
                logger.error(f"Check {check_func.__name__} failed: {e}")
                validation_result['checks_failed'] += 1
                validation_result['overall_valid'] = False
                validation_result['errors'].append(f"Check {check_func.__name__} crashed: {e}")
        
        # Log results
        if validation_result['overall_valid']:
            logger.info(f"✅ {tool_id} output integrity validation passed")
        else:
            logger.error(f"❌ {tool_id} output integrity validation failed")
            logger.error(f"   Errors: {validation_result['errors']}")
        
        return validation_result
    
    def _check_required_keys(self, data: Dict[str, Any], rules: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
        """Check that all required keys are present."""
        
        required_keys = rules.get('required_keys', [])
        missing_keys = []
        
        for key in required_keys:
            if not self._key_exists_nested(data, key):
                missing_keys.append(key)
        
        return {
            'passed': len(missing_keys) == 0,
            'errors': [f"Missing required key: {key}" for key in missing_keys],
            'warnings': [],
            'details': {'required_keys': required_keys, 'missing_keys': missing_keys}
        }
    
    def _check_numeric_validity(self, data: Dict[str, Any], rules: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
        """Check numeric data validity."""
        
        numeric_keys = rules.get('numeric_keys', [])
        errors = []
        warnings = []
        
        for key in numeric_keys:
            value = self._get_nested_value(data, key)
            
            if value is not None:
                if not self._is_numeric_data(value):
                    errors.append(f"Key '{key}' contains non-numeric data")
                elif self._contains_invalid_numbers(value):
                    errors.append(f"Key '{key}' contains invalid numbers (inf, -inf)")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'details': {'checked_keys': numeric_keys}
        }
    
    def _check_array_properties(self, data: Dict[str, Any], rules: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
        """Check array properties (shape, dimensions)."""
        
        array_keys = rules.get('array_keys', [])
        min_shape = rules.get('min_shape', (1,))
        errors = []
        warnings = []
        
        for key in array_keys:
            value = self._get_nested_value(data, key)
            
            if value is not None:
                array_data = self._convert_to_array(value)
                
                if array_data is not None:
                    # Check minimum shape requirements
                    if len(array_data.shape) < len(min_shape):
                        errors.append(f"Array '{key}' has insufficient dimensions")
                    
                    for i, min_dim in enumerate(min_shape):
                        if i < len(array_data.shape) and array_data.shape[i] < min_dim:
                            errors.append(f"Array '{key}' dimension {i} too small: {array_data.shape[i]} < {min_dim}")
                    
                    # Check for empty arrays
                    if array_data.size == 0:
                        errors.append(f"Array '{key}' is empty")
                else:
                    warnings.append(f"Could not convert '{key}' to array")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'details': {'checked_keys': array_keys, 'min_shape': min_shape}
        }
    
    def _check_value_ranges(self, data: Dict[str, Any], rules: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
        """Check that values are within expected ranges."""
        
        value_ranges = rules.get('value_ranges', {})
        errors = []
        warnings = []
        
        for key, (min_val, max_val) in value_ranges.items():
            value = self._get_nested_value(data, key)
            
            if value is not None:
                array_data = self._convert_to_array(value)
                
                if array_data is not None:
                    if np.any(array_data < min_val) or np.any(array_data > max_val):
                        actual_min = np.min(array_data)
                        actual_max = np.max(array_data)
                        errors.append(f"Values in '{key}' outside range [{min_val}, {max_val}]: actual range [{actual_min:.3f}, {actual_max:.3f}]")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'details': {'checked_ranges': value_ranges}
        }
    
    def _check_nan_values(self, data: Dict[str, Any], rules: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
        """Check for NaN values in specified keys."""
        
        no_nan_keys = rules.get('no_nan_keys', [])
        errors = []
        warnings = []
        
        for key in no_nan_keys:
            value = self._get_nested_value(data, key)
            
            if value is not None:
                array_data = self._convert_to_array(value)
                
                if array_data is not None:
                    nan_count = np.sum(np.isnan(array_data))
                    if nan_count > 0:
                        errors.append(f"Key '{key}' contains {nan_count} NaN values")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'details': {'checked_keys': no_nan_keys}
        }
    
    def _check_empty_data(self, data: Dict[str, Any], rules: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
        """Check for empty data structures."""
        
        no_empty_keys = rules.get('no_empty_keys', [])
        errors = []
        warnings = []
        
        for key in no_empty_keys:
            value = self._get_nested_value(data, key)
            
            if value is None:
                errors.append(f"Key '{key}' is None/missing")
            elif self._is_empty_data(value):
                errors.append(f"Key '{key}' contains empty data")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'details': {'checked_keys': no_empty_keys}
        }
    
    def _check_string_validity(self, data: Dict[str, Any], rules: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
        """Check string data validity."""
        
        string_keys = rules.get('string_keys', [])
        min_length = rules.get('min_string_length', 1)
        errors = []
        warnings = []
        
        for key in string_keys:
            value = self._get_nested_value(data, key)
            
            if value is not None:
                if isinstance(value, str):
                    if len(value.strip()) < min_length:
                        errors.append(f"String '{key}' too short: {len(value)} < {min_length}")
                elif isinstance(value, (list, dict)):
                    # Check strings in collections
                    string_values = self._extract_strings_from_collection(value)
                    for i, s in enumerate(string_values):
                        if len(s.strip()) < min_length:
                            errors.append(f"String in '{key}[{i}]' too short")
                else:
                    warnings.append(f"Key '{key}' expected to be string but is {type(value)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'details': {'checked_keys': string_keys, 'min_length': min_length}
        }
    
    def _check_dummy_patterns(self, data: Dict[str, Any], rules: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
        """Check for dummy/placeholder patterns in the data."""
        
        errors = []
        warnings = []
        
        dummy_strings = self._find_dummy_strings(data)
        
        for string_value, location in dummy_strings:
            for pattern in self.dummy_patterns:
                if re.search(pattern, string_value.lower()):
                    errors.append(f"Dummy pattern '{pattern}' found in {location}: '{string_value}'")
                    break
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'details': {'dummy_strings_found': len(dummy_strings)}
        }
    
    def _check_positive_values(self, data: Dict[str, Any], rules: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
        """Check that specified keys contain only positive values."""
        
        positive_keys = rules.get('positive_keys', [])
        errors = []
        warnings = []
        
        for key in positive_keys:
            value = self._get_nested_value(data, key)
            
            if value is not None:
                array_data = self._convert_to_array(value)
                
                if array_data is not None:
                    if np.any(array_data < 0):
                        negative_count = np.sum(array_data < 0)
                        errors.append(f"Key '{key}' contains {negative_count} negative values")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'details': {'checked_keys': positive_keys}
        }
    
    def _check_data_consistency(self, data: Dict[str, Any], rules: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
        """Check for data consistency issues."""
        
        errors = []
        warnings = []
        
        # Check for suspiciously uniform data
        numeric_keys = rules.get('numeric_keys', [])
        
        for key in numeric_keys:
            value = self._get_nested_value(data, key)
            
            if value is not None:
                array_data = self._convert_to_array(value)
                
                if array_data is not None and array_data.size > 1:
                    # Check for constant values (potential dummy data)
                    if np.all(array_data == array_data.flat[0]):
                        warnings.append(f"All values in '{key}' are identical (potential dummy data)")
                    
                    # Check for suspiciously low variance
                    if np.var(array_data) < 1e-10:
                        warnings.append(f"Very low variance in '{key}' (potential dummy data)")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'details': {'consistency_checks': len(numeric_keys)}
        }
    
    # Helper methods
    
    def _key_exists_nested(self, data: Dict[str, Any], key: str) -> bool:
        """Check if a key exists in nested dictionary."""
        try:
            parts = key.split('.')
            current = data
            
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return False
            
            return True
        except:
            return False
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Get value from nested dictionary."""
        try:
            parts = key.split('.')
            current = data
            
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            
            return current
        except:
            return None
    
    def _is_numeric_data(self, value: Any) -> bool:
        """Check if value contains numeric data."""
        try:
            array_data = self._convert_to_array(value)
            return array_data is not None and np.issubdtype(array_data.dtype, np.number)
        except:
            return False
    
    def _contains_invalid_numbers(self, value: Any) -> bool:
        """Check for inf, -inf in numeric data."""
        try:
            array_data = self._convert_to_array(value)
            if array_data is not None:
                return np.any(np.isinf(array_data))
            return False
        except:
            return False
    
    def _convert_to_array(self, value: Any) -> Optional[np.ndarray]:
        """Convert value to numpy array if possible."""
        try:
            if isinstance(value, np.ndarray):
                return value
            elif isinstance(value, (list, tuple)):
                return np.array(value)
            elif isinstance(value, (int, float)):
                return np.array([value])
            else:
                return None
        except:
            return None
    
    def _is_empty_data(self, value: Any) -> bool:
        """Check if data structure is empty."""
        if value is None:
            return True
        elif isinstance(value, (list, tuple, dict, str)):
            return len(value) == 0
        elif isinstance(value, np.ndarray):
            return value.size == 0
        else:
            return False
    
    def _extract_strings_from_collection(self, collection: Union[list, dict]) -> List[str]:
        """Extract string values from collections."""
        strings = []
        
        if isinstance(collection, list):
            for item in collection:
                if isinstance(item, str):
                    strings.append(item)
                elif isinstance(item, (list, dict)):
                    strings.extend(self._extract_strings_from_collection(item))
        
        elif isinstance(collection, dict):
            for value in collection.values():
                if isinstance(value, str):
                    strings.append(value)
                elif isinstance(value, (list, dict)):
                    strings.extend(self._extract_strings_from_collection(value))
        
        return strings
    
    def _find_dummy_strings(self, data: Any, path: str = "root") -> List[Tuple[str, str]]:
        """Find all string values in data structure."""
        dummy_strings = []
        
        if isinstance(data, str):
            dummy_strings.append((data, path))
        
        elif isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}"
                dummy_strings.extend(self._find_dummy_strings(value, new_path))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                dummy_strings.extend(self._find_dummy_strings(item, new_path))
        
        return dummy_strings
    
    def validate_output_file(self, file_path: str, tool_id: str) -> Dict[str, Any]:
        """Validate output from a file."""
        
        try:
            with open(file_path, 'r') as f:
                output_data = json.load(f)
            
            return self.validate_output(output_data, tool_id)
            
        except Exception as e:
            return {
                'tool_id': tool_id,
                'overall_valid': False,
                'checks_passed': 0,
                'checks_failed': 1,
                'errors': [f"Failed to load output file: {e}"],
                'warnings': [],
                'details': {}
            }
    
    def generate_integrity_report(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive integrity report."""
        
        total_tools = len(validation_results)
        valid_tools = sum(1 for r in validation_results if r['overall_valid'])
        total_checks = sum(r['checks_passed'] + r['checks_failed'] for r in validation_results)
        passed_checks = sum(r['checks_passed'] for r in validation_results)
        
        report = {
            'summary': {
                'total_tools_checked': total_tools,
                'valid_tools': valid_tools,
                'integrity_rate': valid_tools / total_tools * 100 if total_tools > 0 else 0,
                'total_checks_run': total_checks,
                'checks_passed': passed_checks,
                'check_success_rate': passed_checks / total_checks * 100 if total_checks > 0 else 0
            },
            'tool_results': validation_results,
            'common_issues': self._identify_common_issues(validation_results),
            'recommendations': self._generate_recommendations(validation_results)
        }
        
        return report
    
    def _identify_common_issues(self, results: List[Dict[str, Any]]) -> List[str]:
        """Identify common issues across tools."""
        
        issue_counts = {}
        
        for result in results:
            for error in result['errors']:
                issue_type = self._categorize_error(error)
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # Return issues that occur in multiple tools
        common_issues = [issue for issue, count in issue_counts.items() 
                        if count > 1]
        
        return common_issues
    
    def _categorize_error(self, error: str) -> str:
        """Categorize error message."""
        error_lower = error.lower()
        
        if 'nan' in error_lower:
            return 'NaN values detected'
        elif 'empty' in error_lower:
            return 'Empty data structures'
        elif 'range' in error_lower:
            return 'Values outside expected range'
        elif 'dummy' in error_lower or 'placeholder' in error_lower:
            return 'Dummy/placeholder data'
        elif 'missing' in error_lower:
            return 'Missing required data'
        else:
            return 'Other validation error'
    
    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation results."""
        
        recommendations = []
        
        # Count different types of issues
        nan_issues = sum(1 for r in results for e in r['errors'] if 'nan' in e.lower())
        empty_issues = sum(1 for r in results for e in r['errors'] if 'empty' in e.lower())
        range_issues = sum(1 for r in results for e in r['errors'] if 'range' in e.lower())
        dummy_issues = sum(1 for r in results for e in r['errors'] if 'dummy' in e.lower())
        
        if nan_issues > 0:
            recommendations.append(f"Address NaN value issues in {nan_issues} tools - add proper numerical validation")
        
        if empty_issues > 0:
            recommendations.append(f"Fix empty data issues in {empty_issues} tools - ensure proper data generation")
        
        if range_issues > 0:
            recommendations.append(f"Fix value range violations in {range_issues} tools - validate output scaling")
        
        if dummy_issues > 0:
            recommendations.append(f"Remove dummy/placeholder data in {dummy_issues} tools - implement proper algorithms")
        
        return recommendations

def main():
    """Main function for running output integrity checks."""
    
    if len(sys.argv) < 3:
        print("Usage: python output_integrity_checker.py <tool_id> <output_file>")
        print("   or: python output_integrity_checker.py --batch <results_directory>")
        sys.exit(1)
    
    checker = OutputIntegrityChecker()
    
    if sys.argv[1] == '--batch':
        # Batch mode - check all files in directory
        results_dir = Path(sys.argv[2])
        
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            sys.exit(1)
        
        validation_results = []
        
        for json_file in results_dir.glob("*.json"):
            # Extract tool_id from filename
            tool_id = json_file.stem.replace('test_output_', '')
            
            result = checker.validate_output_file(str(json_file), tool_id)
            validation_results.append(result)
        
        # Generate comprehensive report
        report = checker.generate_integrity_report(validation_results)
        
        # Save report
        report_file = results_dir / "integrity_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Integrity report saved to {report_file}")
        
        # Print summary
        summary = report['summary']
        print(f"\n🎯 Integrity Check Summary:")
        print(f"   Tools checked: {summary['total_tools_checked']}")
        print(f"   Valid tools: {summary['valid_tools']}")
        print(f"   Integrity rate: {summary['integrity_rate']:.1f}%")
        print(f"   Check success rate: {summary['check_success_rate']:.1f}%")
        
        if summary['integrity_rate'] < 100:
            sys.exit(1)
    
    else:
        # Single file mode
        tool_id = sys.argv[1]
        output_file = sys.argv[2]
        
        result = checker.validate_output_file(output_file, tool_id)
        
        print(json.dumps(result, indent=2, default=str))
        
        if not result['overall_valid']:
            sys.exit(1)

if __name__ == '__main__':
    main()
