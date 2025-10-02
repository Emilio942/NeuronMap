"""Unit tests for data processing modules."""

import unittest
import pytest
import tempfile
import json
import numpy as np
import os
import sys
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.data_processing.quality_manager import DataQualityManager
    from src.data_processing.metadata_manager import MetadataManager, ExperimentTracker
    from src.data_processing.streaming_processor import StreamingDataProcessor, BatchProcessor
except ImportError as e:
    print(f"Warning: Could not import data processing modules: {e}")


class TestDataQualityManager(unittest.TestCase):
    """Test data quality management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_questions = [
            {'text': 'What is artificial intelligence?', 'category': 'factual', 'id': 1},
            {'text': 'What is artificial intelligence?', 'category': 'factual', 'id': 2},  # Duplicate
            {'text': 'How do neural networks learn from data?', 'category': 'reasoning', 'id': 3},
            {'text': 'Hi', 'category': 'factual', 'id': 4},  # Too short
            {'text': '', 'category': 'factual', 'id': 5},  # Empty
            {'text': 'What are the implications of AI for society?', 'category': 'ethical', 'id': 6},
            {'text': 'This is not a question', 'category': 'factual', 'id': 7},  # No question mark
        ]
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_quality_manager_init(self):
        """Test DataQualityManager initialization."""
        try:
            quality_manager = DataQualityManager()
            self.assertIsNotNone(quality_manager)
            
            # Test with custom config
            config = {
                'min_length': 10,
                'max_length': 500,
                'similarity_threshold': 0.8
            }
            quality_manager = DataQualityManager(config)
            self.assertEqual(quality_manager.config['min_length'], 10)
            
        except ImportError:
            self.skipTest("DataQualityManager not available")
    
    def test_duplicate_detection(self):
        """Test duplicate detection functionality."""
        try:
            quality_manager = DataQualityManager()
            
            # Test exact duplicate detection
            duplicates = quality_manager.detect_duplicates(self.test_questions)
            self.assertGreater(len(duplicates), 0)
            
            # Should find the duplicate "What is artificial intelligence?"
            duplicate_texts = [item['text'] for item in duplicates]
            self.assertIn('What is artificial intelligence?', duplicate_texts)
            
            # Test with empty list
            empty_duplicates = quality_manager.detect_duplicates([])
            self.assertEqual(len(empty_duplicates), 0)
            
        except ImportError:
            self.skipTest("DataQualityManager not available")
    
    def test_similarity_detection(self):
        """Test semantic similarity detection."""
        try:
            quality_manager = DataQualityManager()
            
            similar_questions = [
                {'text': 'What is AI?', 'category': 'factual', 'id': 1},
                {'text': 'What is artificial intelligence?', 'category': 'factual', 'id': 2},
                {'text': 'How does machine learning work?', 'category': 'reasoning', 'id': 3}
            ]
            
            similar_pairs = quality_manager.find_similar_questions(
                similar_questions, 
                threshold=0.6
            )
            
            self.assertIsInstance(similar_pairs, list)
            # Should find similarity between "What is AI?" and "What is artificial intelligence?"
            if len(similar_pairs) > 0:
                self.assertIn('similarity_score', similar_pairs[0])
                self.assertIn('question_1', similar_pairs[0])
                self.assertIn('question_2', similar_pairs[0])
            
        except ImportError:
            self.skipTest("DataQualityManager not available")
    
    def test_question_validation(self):
        """Test individual question validation."""
        try:
            quality_manager = DataQualityManager()
            
            # Test validation of each question
            validation_results = []
            for question in self.test_questions:
                is_valid = quality_manager.is_valid_question(question)
                validation_results.append(is_valid)
            
            # Should have some valid and some invalid questions
            valid_count = sum(validation_results)
            invalid_count = len(validation_results) - valid_count
            
            self.assertGreater(valid_count, 0)  # Should have at least one valid question
            self.assertGreater(invalid_count, 0)  # Should have at least one invalid question
            
        except ImportError:
            self.skipTest("DataQualityManager not available")
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        try:
            quality_manager = DataQualityManager()
            
            metrics = quality_manager.calculate_quality_metrics(self.test_questions)
            
            # Check that all expected metrics are present
            expected_metrics = [
                'total_questions', 'valid_questions', 'invalid_questions',
                'duplicate_count', 'average_length', 'quality_score'
            ]
            
            for metric in expected_metrics:
                self.assertIn(metric, metrics)
            
            # Check metric values
            self.assertEqual(metrics['total_questions'], len(self.test_questions))
            self.assertGreaterEqual(metrics['quality_score'], 0)
            self.assertLessEqual(metrics['quality_score'], 100)
            
        except ImportError:
            self.skipTest("DataQualityManager not available")
    
    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        try:
            quality_manager = DataQualityManager()
            
            # Clean the test data
            cleaned_data = quality_manager.clean_data(self.test_questions)
            
            # Cleaned data should have fewer items than original
            self.assertLessEqual(len(cleaned_data), len(self.test_questions))
            
            # All remaining questions should be valid
            for question in cleaned_data:
                self.assertTrue(quality_manager.is_valid_question(question))
            
        except ImportError:
            self.skipTest("DataQualityManager not available")


class TestMetadataManager(unittest.TestCase):
    """Test metadata management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_file = Path(self.temp_dir.name) / "metadata.json"
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_metadata_manager_init(self):
        """Test MetadataManager initialization."""
        try:
            metadata_manager = MetadataManager(str(self.metadata_file))
            self.assertEqual(str(metadata_manager.metadata_file), str(self.metadata_file))
            
        except ImportError:
            self.skipTest("MetadataManager not available")
    
    def test_experiment_recording(self):
        """Test experiment metadata recording."""
        try:
            metadata_manager = MetadataManager(str(self.metadata_file))
            
            # Record an experiment
            experiment_data = {
                'config_name': 'test_config',
                'model_name': 'gpt2',
                'parameters': {
                    'batch_size': 16,
                    'layers': ['layer1', 'layer2'],
                    'num_questions': 100
                },
                'description': 'Test experiment for unit testing'
            }
            
            experiment_id = metadata_manager.record_experiment(**experiment_data)
            
            self.assertIsNotNone(experiment_id)
            self.assertIsInstance(experiment_id, str)
            
            # Verify experiment was recorded
            experiments = metadata_manager.list_experiments()
            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments[0]['config_name'], 'test_config')
            self.assertEqual(experiments[0]['model_name'], 'gpt2')
            
        except ImportError:
            self.skipTest("MetadataManager not available")
    
    def test_experiment_retrieval(self):
        """Test experiment retrieval functionality."""
        try:
            metadata_manager = MetadataManager(str(self.metadata_file))
            
            # Record multiple experiments
            for i in range(3):
                metadata_manager.record_experiment(
                    config_name=f'config_{i}',
                    model_name='gpt2',
                    parameters={'batch_size': 16 + i},
                    description=f'Test experiment {i}'
                )
            
            # Test listing all experiments
            all_experiments = metadata_manager.list_experiments()
            self.assertEqual(len(all_experiments), 3)
            
            # Test filtering experiments
            filtered_experiments = metadata_manager.filter_experiments(
                model_name='gpt2'
            )
            self.assertEqual(len(filtered_experiments), 3)
            
            # Test getting specific experiment
            experiment_id = all_experiments[0]['experiment_id']
            specific_experiment = metadata_manager.get_experiment(experiment_id)
            self.assertIsNotNone(specific_experiment)
            self.assertEqual(specific_experiment['experiment_id'], experiment_id)
            
        except ImportError:
            self.skipTest("MetadataManager not available")
    
    def test_result_recording(self):
        """Test recording experiment results."""
        try:
            metadata_manager = MetadataManager(str(self.metadata_file))
            
            # Record an experiment
            experiment_id = metadata_manager.record_experiment(
                config_name='test_config',
                model_name='gpt2',
                parameters={'batch_size': 16}
            )
            
            # Record results
            results = {
                'activation_statistics': {
                    'mean_activation': 0.5,
                    'std_activation': 0.2,
                    'sparsity': 0.1
                },
                'performance_metrics': {
                    'processing_time': 120.5,
                    'memory_used_mb': 2048
                },
                'output_files': ['activations.csv', 'plots.png']
            }
            
            metadata_manager.record_results(experiment_id, results)
            
            # Verify results were recorded
            experiment = metadata_manager.get_experiment(experiment_id)
            self.assertIn('results', experiment)
            self.assertEqual(
                experiment['results']['activation_statistics']['mean_activation'], 
                0.5
            )
            
        except ImportError:
            self.skipTest("MetadataManager not available")
    
    def test_metadata_persistence(self):
        """Test metadata persistence across instances."""
        try:
            # Create first instance and record data
            metadata_manager1 = MetadataManager(str(self.metadata_file))
            experiment_id = metadata_manager1.record_experiment(
                config_name='persistent_test',
                model_name='bert',
                parameters={'batch_size': 8}
            )
            
            # Create second instance and verify data persists
            metadata_manager2 = MetadataManager(str(self.metadata_file))
            experiments = metadata_manager2.list_experiments()
            
            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments[0]['config_name'], 'persistent_test')
            self.assertEqual(experiments[0]['experiment_id'], experiment_id)
            
        except ImportError:
            self.skipTest("MetadataManager not available")


class TestExperimentTracker(unittest.TestCase):
    """Test experiment tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_experiment_tracker_init(self):
        """Test ExperimentTracker initialization."""
        try:
            tracker = ExperimentTracker(base_dir=self.temp_dir.name)
            self.assertEqual(str(tracker.base_dir), self.temp_dir.name)
            
        except ImportError:
            self.skipTest("ExperimentTracker not available")
    
    def test_experiment_lifecycle(self):
        """Test complete experiment lifecycle tracking."""
        try:
            tracker = ExperimentTracker(base_dir=self.temp_dir.name)
            
            # Start experiment
            experiment_id = tracker.start_experiment(
                name='lifecycle_test',
                config={'model': 'gpt2', 'batch_size': 16}
            )
            
            # Log progress
            tracker.log_progress(experiment_id, 'Data generation', 25)
            tracker.log_progress(experiment_id, 'Activation extraction', 50)
            tracker.log_progress(experiment_id, 'Analysis', 75)
            tracker.log_progress(experiment_id, 'Visualization', 100)
            
            # Finish experiment
            results = {'accuracy': 0.95, 'loss': 0.05}
            tracker.finish_experiment(experiment_id, results)
            
            # Verify experiment status
            experiment = tracker.get_experiment_status(experiment_id)
            self.assertEqual(experiment['status'], 'completed')
            self.assertEqual(experiment['progress'], 100)
            self.assertIn('results', experiment)
            
        except ImportError:
            self.skipTest("ExperimentTracker not available")


class TestStreamingDataProcessor(unittest.TestCase):
    """Test streaming data processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test data file
        self.test_data = []
        for i in range(1000):
            self.test_data.append({
                'question': f'Question {i}?',
                'category': 'factual' if i % 2 == 0 else 'reasoning',
                'activations': np.random.randn(768).tolist()
            })
        
        self.data_file = Path(self.temp_dir.name) / "test_data.json"
        with open(self.data_file, 'w') as f:
            json.dump(self.test_data, f)
            
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_streaming_processor_init(self):
        """Test StreamingDataProcessor initialization."""
        try:
            config = {
                'batch_size': 32,
                'buffer_size': 1024,
                'num_workers': 2
            }
            processor = StreamingDataProcessor(config)
            self.assertEqual(processor.config['batch_size'], 32)
            
        except ImportError:
            self.skipTest("StreamingDataProcessor not available")
    
    def test_data_streaming(self):
        """Test streaming data processing."""
        try:
            config = {
                'batch_size': 10,
                'buffer_size': 100
            }
            processor = StreamingDataProcessor(config)
            
            # Process data in batches
            processed_batches = []
            for batch in processor.stream_data(str(self.data_file)):
                processed_batches.append(batch)
                if len(processed_batches) >= 5:  # Process first 5 batches
                    break
            
            self.assertGreater(len(processed_batches), 0)
            
            # Each batch should have the correct size (except possibly the last one)
            for i, batch in enumerate(processed_batches[:-1]):
                self.assertEqual(len(batch), config['batch_size'])
            
        except ImportError:
            self.skipTest("StreamingDataProcessor not available")
    
    def test_memory_efficient_processing(self):
        """Test memory-efficient data processing."""
        try:
            config = {
                'batch_size': 50,
                'memory_limit_mb': 100  # Small memory limit
            }
            processor = StreamingDataProcessor(config)
            
            # Process with memory constraints
            total_processed = 0
            for batch in processor.stream_data_memory_efficient(str(self.data_file)):
                total_processed += len(batch)
                if total_processed >= 200:  # Process first 200 items
                    break
            
            self.assertGreaterEqual(total_processed, 200)
            
        except ImportError:
            self.skipTest("StreamingDataProcessor not available")


class TestBatchProcessor(unittest.TestCase):
    """Test batch processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_activations = {
            'layer_0': np.random.randn(100, 768),
            'layer_6': np.random.randn(100, 768),
            'layer_11': np.random.randn(100, 768)
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_batch_processor_init(self):
        """Test BatchProcessor initialization."""
        try:
            config = {
                'batch_size': 16,
                'num_workers': 4,
                'parallel_processing': True
            }
            processor = BatchProcessor(config)
            self.assertEqual(processor.config['batch_size'], 16)
            
        except ImportError:
            self.skipTest("BatchProcessor not available")
    
    def test_activation_batch_processing(self):
        """Test batch processing of activation data."""
        try:
            config = {'batch_size': 20}
            processor = BatchProcessor(config)
            
            # Process activations in batches
            processed_results = []
            for layer_name, activations in self.test_activations.items():
                results = processor.process_activation_batches(
                    activations,
                    processing_func=lambda x: np.mean(x, axis=1)  # Calculate mean per sample
                )
                processed_results.append((layer_name, results))
            
            self.assertEqual(len(processed_results), 3)  # Three layers
            
            # Check results shape
            for layer_name, results in processed_results:
                self.assertEqual(len(results), 100)  # 100 samples
                
        except ImportError:
            self.skipTest("BatchProcessor not available")
    
    def test_parallel_processing(self):
        """Test parallel batch processing."""
        try:
            config = {
                'batch_size': 25,
                'num_workers': 2,
                'parallel_processing': True
            }
            processor = BatchProcessor(config)
            
            # Process with parallel workers
            processing_times = []
            
            # Sequential processing
            start_time = time.time()
            sequential_results = processor.process_activation_batches(
                self.test_activations['layer_0'],
                processing_func=lambda x: np.std(x, axis=1),
                parallel=False
            )
            sequential_time = time.time() - start_time
            processing_times.append(('sequential', sequential_time))
            
            # Parallel processing
            start_time = time.time()
            parallel_results = processor.process_activation_batches(
                self.test_activations['layer_0'],
                processing_func=lambda x: np.std(x, axis=1),
                parallel=True
            )
            parallel_time = time.time() - start_time
            processing_times.append(('parallel', parallel_time))
            
            # Results should be the same
            np.testing.assert_array_almost_equal(sequential_results, parallel_results)
            
            # Print timing comparison (parallel might not always be faster for small data)
            print(f"Sequential: {sequential_time:.4f}s, Parallel: {parallel_time:.4f}s")
            
        except ImportError:
            self.skipTest("BatchProcessor not available")


class TestDataProcessingIntegration(unittest.TestCase):
    """Integration tests for data processing modules."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        """Clean up integration test environment."""
        self.temp_dir.cleanup()
    
    def test_full_data_processing_pipeline(self):
        """Test complete data processing pipeline."""
        try:
            # Create test data
            questions = [
                {'text': f'Question {i}?', 'category': 'factual', 'id': i}
                for i in range(50)
            ]
            
            # Add some problematic data
            questions.extend([
                {'text': 'Question 1?', 'category': 'factual', 'id': 99},  # Duplicate
                {'text': '', 'category': 'factual', 'id': 100},  # Empty
                {'text': 'Hi', 'category': 'factual', 'id': 101},  # Too short
            ])
            
            # Step 1: Quality management
            quality_manager = DataQualityManager()
            cleaned_questions = quality_manager.clean_data(questions)
            quality_metrics = quality_manager.calculate_quality_metrics(cleaned_questions)
            
            # Step 2: Metadata management
            metadata_file = Path(self.temp_dir.name) / "metadata.json"
            metadata_manager = MetadataManager(str(metadata_file))
            
            experiment_id = metadata_manager.record_experiment(
                config_name='integration_test',
                model_name='test_model',
                parameters={
                    'original_questions': len(questions),
                    'cleaned_questions': len(cleaned_questions),
                    'quality_score': quality_metrics['quality_score']
                }
            )
            
            # Step 3: Streaming processing
            processor_config = {'batch_size': 10}
            streaming_processor = StreamingDataProcessor(processor_config)
            
            # Save cleaned data for streaming
            data_file = Path(self.temp_dir.name) / "cleaned_data.json"
            with open(data_file, 'w') as f:
                json.dump(cleaned_questions, f)
            
            # Process in batches
            processed_count = 0
            for batch in streaming_processor.stream_data(str(data_file)):
                processed_count += len(batch)
                if processed_count >= 30:  # Process first 30 items
                    break
            
            # Step 4: Record final results
            final_results = {
                'original_count': len(questions),
                'cleaned_count': len(cleaned_questions),
                'processed_count': processed_count,
                'quality_metrics': quality_metrics
            }
            
            metadata_manager.record_results(experiment_id, final_results)
            
            # Verify the complete pipeline
            experiment = metadata_manager.get_experiment(experiment_id)
            self.assertIn('results', experiment)
            self.assertEqual(experiment['results']['original_count'], len(questions))
            self.assertLess(experiment['results']['cleaned_count'], len(questions))
            self.assertGreater(experiment['results']['processed_count'], 0)
            
        except ImportError:
            self.skipTest("Data processing modules not available")


if __name__ == "__main__":
    unittest.main(verbosity=2)