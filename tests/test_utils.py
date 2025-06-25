"""Unit tests for utility modules."""

import unittest
import pytest
import tempfile
import json
import os
import sys
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.utils.error_handling import NeuronMapError, ValidationError, ErrorHandler
    from src.utils.file_handlers import FileManager, DataFileHandler
    from src.utils.monitoring import PerformanceMonitor, ResourceMonitor
    from src.utils.performance import PerformanceOptimizer, MemoryOptimizer
    from src.utils.validation import DataValidator, ConfigValidator
except ImportError as e:
    print(f"Warning: Could not import utility modules: {e}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling functionality."""
    
    def test_custom_exceptions(self):
        """Test custom exception types."""
        try:
            # Test NeuronMapError
            with self.assertRaises(NeuronMapError):
                raise NeuronMapError("Test error message")
            
            # Test ValidationError
            with self.assertRaises(ValidationError):
                raise ValidationError("Validation failed")
            
            # Test exception inheritance
            with self.assertRaises(Exception):
                raise NeuronMapError("Should be caught as general Exception")
                
        except ImportError:
            self.skipTest("Custom exceptions not available")
    
    def test_error_handler(self):
        """Test ErrorHandler functionality."""
        try:
            handler = ErrorHandler()
            
            # Test error recording
            handler.record_error("TestError", "Test error message", {"context": "test"})
            
            errors = handler.get_errors()
            self.assertGreater(len(errors), 0)
            self.assertEqual(errors[0]['error_type'], "TestError")
            self.assertEqual(errors[0]['message'], "Test error message")
            
        except ImportError:
            self.skipTest("ErrorHandler not available")
    
    def test_error_context_manager(self):
        """Test error handling context manager."""
        try:
            with ErrorHandler() as handler:
                # Simulate an error
                try:
                    raise ValueError("Test error")
                except ValueError as e:
                    handler.record_error("ValueError", str(e))
            
            errors = handler.get_errors()
            self.assertEqual(len(errors), 1)
            
        except ImportError:
            self.skipTest("ErrorHandler not available")


class TestFileHandlers(unittest.TestCase):
    """Test file handling utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_data = {
            'questions': ['Q1', 'Q2', 'Q3'],
            'activations': {'layer_0': [[1, 2, 3], [4, 5, 6]]},
            'metadata': {'model': 'test'}
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_file_manager_basic_operations(self):
        """Test basic file manager operations."""
        try:
            file_manager = FileManager(base_dir=self.temp_dir.name)
            
            # Test file writing
            test_file = "test_data.json"
            file_manager.write_json(test_file, self.test_data)
            
            # Test file reading
            loaded_data = file_manager.read_json(test_file)
            self.assertEqual(loaded_data['questions'], self.test_data['questions'])
            
            # Test file existence check
            self.assertTrue(file_manager.file_exists(test_file))
            self.assertFalse(file_manager.file_exists("nonexistent.json"))
            
        except ImportError:
            self.skipTest("FileManager not available")
    
    def test_data_file_handler(self):
        """Test specialized data file handling."""
        try:
            handler = DataFileHandler(data_dir=self.temp_dir.name)
            
            # Test CSV handling
            csv_data = [
                {'question': 'Q1', 'activation': 0.5},
                {'question': 'Q2', 'activation': 0.8}
            ]
            csv_file = "test_data.csv"
            handler.write_csv(csv_file, csv_data)
            
            loaded_csv = handler.read_csv(csv_file)
            self.assertEqual(len(loaded_csv), 2)
            
            # Test HDF5 handling (if available)
            try:
                import h5py
                import numpy as np
                
                hdf5_data = {'activations': np.random.randn(10, 768)}
                hdf5_file = "test_data.h5"
                handler.write_hdf5(hdf5_file, hdf5_data)
                
                loaded_hdf5 = handler.read_hdf5(hdf5_file)
                self.assertIn('activations', loaded_hdf5)
                
            except ImportError:
                print("h5py not available, skipping HDF5 tests")
            
        except ImportError:
            self.skipTest("DataFileHandler not available")
    
    def test_file_compression(self):
        """Test file compression utilities."""
        try:
            file_manager = FileManager(base_dir=self.temp_dir.name)
            
            # Create a file to compress
            large_data = {'data': list(range(1000))}
            original_file = "large_data.json"
            file_manager.write_json(original_file, large_data)
            
            # Test compression
            compressed_file = file_manager.compress_file(original_file)
            self.assertTrue(file_manager.file_exists(compressed_file))
            
            # Test decompression
            decompressed_file = file_manager.decompress_file(compressed_file)
            loaded_data = file_manager.read_json(decompressed_file)
            self.assertEqual(len(loaded_data['data']), 1000)
            
        except ImportError:
            self.skipTest("File compression not available")


class TestMonitoring(unittest.TestCase):
    """Test monitoring functionality."""  
    
    def test_performance_monitor(self):
        """Test performance monitoring."""
        try:
            monitor = PerformanceMonitor()
            
            # Test timing context manager
            with monitor.time_operation("test_operation"):
                time.sleep(0.1)  # Simulate work
            
            metrics = monitor.get_metrics()
            self.assertIn("test_operation", metrics)
            self.assertGreater(metrics["test_operation"]["duration"], 0.05)  # Should be ~0.1s
            
        except ImportError:
            self.skipTest("PerformanceMonitor not available")
    
    def test_resource_monitor(self):
        """Test resource monitoring."""
        try:
            monitor = ResourceMonitor()
            
            # Test memory monitoring
            memory_info = monitor.get_memory_usage()
            self.assertIn("current_mb", memory_info)
            self.assertIn("peak_mb", memory_info)
            self.assertGreater(memory_info["current_mb"], 0)
            
            # Test GPU monitoring (if available)
            try:
                gpu_info = monitor.get_gpu_usage()
                if gpu_info:
                    self.assertIn("gpu_memory_mb", gpu_info)
                    self.assertIn("gpu_utilization", gpu_info)
            except:
                print("GPU monitoring not available")
            
        except ImportError:
            self.skipTest("ResourceMonitor not available")
    
    def test_monitoring_integration(self):
        """Test integrated monitoring."""
        try:
            perf_monitor = PerformanceMonitor()
            resource_monitor = ResourceMonitor()
            
            # Start monitoring
            with perf_monitor.time_operation("test_with_resources"):
                initial_memory = resource_monitor.get_memory_usage()["current_mb"]
                
                # Simulate memory allocation
                large_list = [0] * 100000
                
                final_memory = resource_monitor.get_memory_usage()["current_mb"]
                
                # Memory should have increased
                self.assertGreaterEqual(final_memory, initial_memory)
            
            # Check performance metrics
            metrics = perf_monitor.get_metrics()
            self.assertIn("test_with_resources", metrics)
            
        except ImportError:
            self.skipTest("Monitoring modules not available")


class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization utilities."""
    
    def test_performance_optimizer(self):
        """Test performance optimization."""
        try:
            optimizer = PerformanceOptimizer()
            
            # Test batch size optimization
            optimal_batch_size = optimizer.find_optimal_batch_size(
                data_size=1000,
                memory_limit_mb=512
            )
            self.assertIsInstance(optimal_batch_size, int)
            self.assertGreater(optimal_batch_size, 0)
            
            # Test memory estimation
            memory_estimate = optimizer.estimate_memory_usage(
                batch_size=32,
                sequence_length=512,
                hidden_size=768
            )
            self.assertIsInstance(memory_estimate, (int, float))
            self.assertGreater(memory_estimate, 0)
            
        except ImportError:
            self.skipTest("PerformanceOptimizer not available")
    
    def test_memory_optimizer(self):
        """Test memory optimization."""
        try:
            optimizer = MemoryOptimizer()
            
            # Test garbage collection optimization
            initial_memory = optimizer.get_current_memory_mb()
            
            # Force garbage collection
            optimizer.force_garbage_collection()
            
            # Test memory cleanup
            optimizer.cleanup_memory()
            
            final_memory = optimizer.get_current_memory_mb()
            # Memory should be the same or lower after cleanup
            self.assertLessEqual(final_memory, initial_memory + 10)  # Allow small variance
            
        except ImportError:
            self.skipTest("MemoryOptimizer not available")
    
    @patch('torch.cuda.is_available')
    def test_gpu_optimization(self, mock_cuda_available):
        """Test GPU optimization utilities."""
        try:
            # Mock CUDA availability
            mock_cuda_available.return_value = True
            
            optimizer = PerformanceOptimizer()
            
            # Test GPU memory estimation
            gpu_memory = optimizer.estimate_gpu_memory_usage(
                model_size_mb=500,
                batch_size=16,
                sequence_length=512
            )
            self.assertIsInstance(gpu_memory, (int, float))
            self.assertGreater(gpu_memory, 0)
            
        except ImportError:
            self.skipTest("GPU optimization not available")


class TestValidation(unittest.TestCase):
    """Test validation utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_config = {
            'model': 'gpt2',
            'batch_size': 16,
            'layers': ['layer1', 'layer2'],
            'output_dir': '/tmp/test'
        }
        
        self.invalid_config = {
            'model': '',  # Empty model name
            'batch_size': -1,  # Invalid batch size
            'layers': [],  # Empty layers list
            'output_dir': None  # Invalid output dir
        }
    
    def test_config_validator(self):
        """Test configuration validation."""
        try:
            validator = ConfigValidator()
            
            # Test valid config
            is_valid, errors = validator.validate_config(self.valid_config)
            self.assertTrue(is_valid)
            self.assertEqual(len(errors), 0)
            
            # Test invalid config
            is_valid, errors = validator.validate_config(self.invalid_config)
            self.assertFalse(is_valid)
            self.assertGreater(len(errors), 0)
            
            # Check specific error types
            error_types = [error['type'] for error in errors]
            self.assertIn('invalid_model', error_types)
            self.assertIn('invalid_batch_size', error_types)
            
        except ImportError:
            self.skipTest("ConfigValidator not available")
    
    def test_data_validator(self):
        """Test data validation."""
        try:
            validator = DataValidator()
            
            # Test valid data
            valid_data = {
                'questions': ['Q1?', 'Q2?', 'Q3?'],
                'activations': {'layer_0': [[1, 2], [3, 4], [5, 6]]},
                'metadata': {'model': 'test'}
            }
            
            is_valid, errors = validator.validate_data(valid_data)
            self.assertTrue(is_valid)
            
            # Test invalid data
            invalid_data = {
                'questions': ['', 'Q2?'],  # Empty question
                'activations': {'layer_0': [[1, 2], [3]]},  # Inconsistent shapes
                'metadata': {}  # Missing model info
            }
            
            is_valid, errors = validator.validate_data(invalid_data)
            self.assertFalse(is_valid)
            self.assertGreater(len(errors), 0)
            
        except ImportError:
            self.skipTest("DataValidator not available")
    
    def test_system_requirements_validation(self):
        """Test system requirements validation."""
        try:
            from src.utils.validation import check_system_requirements
            
            requirements = check_system_requirements()
            
            # Should return status for key packages
            expected_packages = ['numpy', 'pandas', 'scikit-learn']
            for package in expected_packages:
                self.assertIn(package, requirements)
                self.assertIn('available', requirements[package])
                self.assertIn('version', requirements[package])
            
        except ImportError:
            self.skipTest("System requirements check not available")


class TestUtilsIntegration(unittest.TestCase):
    """Integration tests for utility modules."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_config = {
            'model': 'test-model',
            'batch_size': 8,
            'output_dir': self.temp_dir.name
        }
        
    def tearDown(self):
        """Clean up integration test environment."""
        self.temp_dir.cleanup()
    
    def test_error_handling_with_file_operations(self):
        """Test error handling during file operations."""
        try:
            error_handler = ErrorHandler()
            file_manager = FileManager(base_dir=self.temp_dir.name)
            
            # Test file operation with error handling
            try:
                # Try to read non-existent file
                file_manager.read_json("nonexistent.json")
            except Exception as e:
                error_handler.record_error("FileNotFound", str(e))
            
            errors = error_handler.get_errors()
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]['error_type'], "FileNotFound")
            
        except ImportError:
            self.skipTest("Integration modules not available")
    
    def test_performance_monitoring_with_file_operations(self):
        """Test performance monitoring during file operations."""
        try:
            monitor = PerformanceMonitor()
            file_manager = FileManager(base_dir=self.temp_dir.name)
            
            # Monitor file operations
            with monitor.time_operation("file_write"):
                large_data = {'data': list(range(10000))}
                file_manager.write_json("large_file.json", large_data)
            
            with monitor.time_operation("file_read"):
                loaded_data = file_manager.read_json("large_file.json")
            
            metrics = monitor.get_metrics()
            self.assertIn("file_write", metrics)
            self.assertIn("file_read", metrics)
            
            # Write should typically be slower than read
            write_time = metrics["file_write"]["duration"]
            read_time = metrics["file_read"]["duration"]
            self.assertGreater(write_time, 0)
            self.assertGreater(read_time, 0)
            
        except ImportError:
            self.skipTest("Integration modules not available")


class MockDependencyTests(unittest.TestCase):
    """Test utility functions with mocked dependencies."""
    
    def test_torch_unavailable(self):
        """Test behavior when PyTorch is unavailable."""
        try:
            with patch.dict('sys.modules', {'torch': None}):
                # Should handle missing torch gracefully
                from src.utils.performance import PerformanceOptimizer
                
                optimizer = PerformanceOptimizer()
                
                # GPU operations should return safe defaults
                memory_estimate = optimizer.estimate_gpu_memory_usage(
                    model_size_mb=500,
                    batch_size=16,
                    sequence_length=512
                )
                
                # Should return a reasonable estimate even without torch
                self.assertIsInstance(memory_estimate, (int, float))
                
        except ImportError:
            self.skipTest("PerformanceOptimizer not available")
    
    def test_psutil_unavailable(self):
        """Test behavior when psutil is unavailable."""
        try:
            with patch.dict('sys.modules', {'psutil': None}):
                # Should handle missing psutil gracefully
                from src.utils.monitoring import ResourceMonitor
                
                monitor = ResourceMonitor()
                
                # Should provide fallback memory monitoring
                memory_info = monitor.get_memory_usage()
                self.assertIsInstance(memory_info, dict)
                
        except ImportError:
            self.skipTest("ResourceMonitor not available")


if __name__ == "__main__":
    unittest.main(verbosity=2)