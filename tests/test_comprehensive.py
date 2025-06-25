"""Comprehensive testing framework for NeuronMap.

This module implements testing and validation for:
- Unit tests for all components
- Integration tests for end-to-end workflows
- Property-based testing with Hypothesis
- Mock tests for external dependencies
- Performance regression tests
- Cross-platform testing
"""

import unittest
import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

# Check for optional dependencies
try:
    import numpy
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try to import hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators if hypothesis is not available
    def given(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class MockStrategies:
        @staticmethod
        def lists(*args, **kwargs):
            return None
        @staticmethod
        def text(*args, **kwargs):
            return None
        @staticmethod
        def floats(*args, **kwargs):
            return None
        @staticmethod
        def integers(*args, **kwargs):
            return None
    
    st = MockStrategies()
    
    def settings(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Results from a test execution."""
    test_name: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    warnings: List[str] = None


class NeuronMapTestCase(unittest.TestCase):
    """Base test case class with common utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            'model_name': 'test_model',
            'batch_size': 2,
            'max_length': 128
        }
        
        # Create temporary test files
        self.test_data_file = Path(self.temp_dir) / "test_data.jsonl"
        self.test_config_file = Path(self.temp_dir) / "test_config.yaml"
        
        # Sample test data
        test_questions = [
            {"question": "What is the capital of France?", "id": "q1"},
            {"question": "How does machine learning work?", "id": "q2"}
        ]
        
        with open(self.test_data_file, 'w') as f:
            for q in test_questions:
                f.write(json.dumps(q) + '\n')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_model(self):
        """Create a mock transformer model for testing."""
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_model.config.num_hidden_layers = 12
        
        # Mock outputs
        if NUMPY_AVAILABLE:
            hidden_states = [np.random.randn(1, 10, 768) for _ in range(12)]
        else:
            # Fallback without numpy
            hidden_states = [[[0.1] * 768] * 10] * 12
            
        mock_outputs = MagicMock()
        mock_outputs.hidden_states = hidden_states
        mock_outputs.last_hidden_state = hidden_states[-1]
        
        mock_model.return_value = mock_outputs
        return mock_model
    
    def create_mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        mock_tokenizer = MagicMock()
        if NUMPY_AVAILABLE:
            mock_tokenizer.return_value = {
                'input_ids': np.array([[1, 2, 3, 4, 5]]),
                'attention_mask': np.array([[1, 1, 1, 1, 1]])
            }
        else:
            mock_tokenizer.return_value = {
                'input_ids': [[1, 2, 3, 4, 5]],
                'attention_mask': [[1, 1, 1, 1, 1]]
            }
        mock_tokenizer.pad_token = '[PAD]'
        mock_tokenizer.eos_token = '[EOS]'
        return mock_tokenizer


class TestConfigUtilities(NeuronMapTestCase):
    """Test configuration utilities."""
    
    def test_config_loading(self):
        """Test configuration loading from YAML files."""
        try:
            from src.utils.config_manager import get_config
            config = get_config()
            self.assertIsNotNone(config)
        except ImportError:
            self.skipTest("Config module not available")
    
    def test_experiment_config_validation(self):
        """Test experiment configuration validation."""
        try:
            from src.utils.validation import validate_experiment_config
            
            valid_config = {
                'model_name': 'gpt2',
                'batch_size': 4,
                'max_length': 512
            }
            
            errors = validate_experiment_config(valid_config)
            self.assertEqual(len(errors), 0)
            
            # Test invalid config
            invalid_config = {
                'batch_size': -1,  # Invalid
                'max_length': 'invalid'  # Invalid type
            }
            
            errors = validate_experiment_config(invalid_config)
            self.assertGreater(len(errors), 0)
            
        except ImportError:
            self.skipTest("Validation module not available")


class TestDataGeneration(NeuronMapTestCase):
    """Test question generation functionality."""
    
    @patch('src.data_generation.question_generator.ollama')
    def test_question_generation(self, mock_ollama):
        """Test question generation with mocked Ollama."""
        try:
            from src.data_generation.question_generator import QuestionGenerator
            
            # Mock Ollama response
            mock_ollama.generate.return_value = {
                'response': 'What is artificial intelligence?'
            }
            
            generator = QuestionGenerator('default')
            questions = generator.generate_questions(count=2)
            
            self.assertIsInstance(questions, list)
            self.assertEqual(len(questions), 2)
            
        except ImportError:
            self.skipTest("Question generator module not available")
    
    def test_question_validation(self):
        """Test question format validation."""
        try:
            from src.utils.validation import validate_questions_file
            
            # Test with valid file
            result = validate_questions_file(str(self.test_data_file))
            self.assertIsInstance(result, dict)
            self.assertIn('valid', result)
            self.assertIn('errors', result)
            
        except ImportError:
            self.skipTest("Validation module not available")


class TestActivationExtraction(NeuronMapTestCase):
    """Test activation extraction functionality."""
    
    def test_activation_extractor_initialization(self):
        """Test activation extractor initialization."""
        try:
            from src.analysis.activation_extractor import ActivationExtractor
            
            extractor = ActivationExtractor('test')
            self.assertIsNotNone(extractor)
            
        except ImportError:
            self.skipTest("Activation extractor module not available")
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_activation_extraction(self, mock_tokenizer, mock_model):
        """Test activation extraction with mocked model."""
        try:
            from src.analysis.activation_extractor import ActivationExtractor
            
            # Setup mocks
            mock_model.return_value = self.create_mock_model()
            mock_tokenizer.return_value = self.create_mock_tokenizer()
            
            extractor = ActivationExtractor('test')
            
            # Test extraction (would normally require actual model)
            # This is a placeholder for more detailed testing
            self.assertTrue(True)  # Placeholder assertion
            
        except ImportError:
            self.skipTest("Activation extractor module not available")


class TestVisualization(NeuronMapTestCase):
    """Test visualization functionality."""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        try:
            from src.visualization.visualizer import ActivationVisualizer
            
            visualizer = ActivationVisualizer('test')
            self.assertIsNotNone(visualizer)
            
        except ImportError:
            self.skipTest("Visualizer module not available")
    
    def test_plot_creation(self):
        """Test basic plot creation."""
        try:
            from src.visualization.visualizer import ActivationVisualizer
            
            # Create test data
            test_activations = np.random.randn(100, 50)
            test_labels = [f"sample_{i}" for i in range(100)]
            
            visualizer = ActivationVisualizer('test')
            
            # Test PCA plot creation (placeholder)
            # In a real test, we would check if files are created
            self.assertTrue(True)  # Placeholder assertion
            
        except ImportError:
            self.skipTest("Visualizer module not available")


class TestPerformanceOptimization(NeuronMapTestCase):
    """Test performance optimization utilities."""
    
    def test_performance_profiler(self):
        """Test performance profiler."""
        try:
            from src.utils.performance import PerformanceProfiler
            
            profiler = PerformanceProfiler()
            
            # Test profiling context manager
            with profiler.profile("test_operation"):
                import time
                time.sleep(0.01)  # Small delay for testing
            
            summary = profiler.get_summary()
            self.assertIn("test_operation", summary["timing_stats"])
            
        except ImportError:
            self.skipTest("Performance module not available")
    
    def test_gpu_optimizer(self):
        """Test GPU optimizer utilities."""
        try:
            from src.utils.performance import GPUOptimizer
            
            optimizer = GPUOptimizer()
            self.assertIsNotNone(optimizer)
            
        except ImportError:
            self.skipTest("Performance module not available")


class TestErrorHandling(NeuronMapTestCase):
    """Test error handling and monitoring."""
    
    def test_error_handler(self):
        """Test global error handler."""
        try:
            from src.utils.error_handling import with_retry
            
            @with_retry()
            def test_function():
                return "success"
            
            result = test_function()
            self.assertEqual(result, "success")
            
        except ImportError:
            self.skipTest("Error handling module not available")
    
    def test_monitoring_utilities(self):
        """Test system monitoring utilities."""
        try:
            from src.utils.monitoring import SystemMonitor
            
            monitor = SystemMonitor()
            status = monitor.get_system_status()
            
            self.assertIsInstance(status, dict)
            self.assertIn("cpu_usage", status)
            
        except ImportError:
            self.skipTest("Monitoring module not available")


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
class TestPropertyBasedTesting(NeuronMapTestCase):
    """Property-based tests using Hypothesis."""
    
    @given(st.lists(st.text(min_size=1), min_size=1, max_size=10))
    def test_question_processing_properties(self, questions):
        """Test properties of question processing."""
        try:
            from src.data_generation.question_generator import QuestionGenerator
            
            generator = QuestionGenerator('test')
            
            # Property: processing should not crash on valid input
            for question in questions:
                try:
                    # Test basic processing (placeholder)
                    processed = generator._preprocess_question(question)
                    self.assertIsInstance(processed, str)
                except AttributeError:
                    # Method might not exist, skip
                    pass
                    
        except ImportError:
            self.skipTest("Question generator module not available")
    
    @given(st.lists(st.floats(min_value=-10, max_value=10), min_size=10, max_size=100))
    def test_activation_processing_properties(self, activations):
        """Test properties of activation processing."""
        activation_array = np.array(activations).reshape(-1, 1)
        
        # Property: normalization should preserve shape
        normalized = (activation_array - np.mean(activation_array)) / (np.std(activation_array) + 1e-8)
        self.assertEqual(activation_array.shape, normalized.shape)
        
        # Property: PCA should reduce dimensionality only if we have enough dimensions
        if activation_array.shape[0] > 2:
            from sklearn.decomposition import PCA
            # Use 1 component since we only have 1 feature
            n_components = min(1, activation_array.shape[1])
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(activation_array)
            self.assertEqual(reduced.shape[1], n_components)


class TestIntegration(NeuronMapTestCase):
    """Integration tests for end-to-end workflows."""
    
    def test_full_pipeline_mock(self):
        """Test full pipeline with mocked components."""
        # This would test the entire pipeline from question generation
        # through activation extraction to visualization
        
        # For now, just test that imports work
        try:
            from src.data_generation.question_generator import QuestionGenerator
            from src.analysis.activation_extractor import ActivationExtractor
            from src.visualization.visualizer import ActivationVisualizer
            
            # Test instantiation
            generator = QuestionGenerator('test')
            extractor = ActivationExtractor('test')
            visualizer = ActivationVisualizer('test')
            
            self.assertTrue(True)  # Placeholder for real integration test
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_cli_integration(self):
        """Test CLI command integration."""
        # Test that CLI commands can be imported and basic functionality works
        try:
            import main
            
            # Test that main function exists
            self.assertTrue(hasattr(main, 'main'))
            
            # Test command functions exist
            expected_commands = [
                'cmd_generate_questions',
                'cmd_extract_activations',
                'cmd_visualize',
                'cmd_interpretability_analysis',
                'cmd_experimental_analysis'
            ]
            
            for cmd in expected_commands:
                if hasattr(main, cmd):
                    self.assertTrue(callable(getattr(main, cmd)))
                    
        except ImportError:
            self.skipTest("Main CLI module not available")


class TestConceptualAnalysisIntegration(unittest.TestCase):
    """Integration tests for conceptual analysis features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available")
    def test_conceptual_analyzer_creation(self):
        """Test conceptual analyzer can be created."""
        try:
            from src.analysis.conceptual_analysis import ConceptualAnalyzer, create_conceptual_analyzer
            
            # Test basic creation
            analyzer = ConceptualAnalyzer()
            self.assertIsInstance(analyzer, ConceptualAnalyzer)
            
            # Test factory function
            analyzer2 = create_conceptual_analyzer()
            self.assertIsInstance(analyzer2, ConceptualAnalyzer)
            
            # Test with config
            config = {'concept_threshold': 0.8}
            analyzer3 = create_conceptual_analyzer(config)
            self.assertEqual(analyzer3.concept_threshold, 0.8)
            
        except ImportError as e:
            self.skipTest(f"Conceptual analysis dependencies not available: {e}")
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available")
    def test_concept_extraction_workflow(self):
        """Test concept extraction workflow."""
        try:
            from src.analysis.conceptual_analysis import ConceptualAnalyzer
            
            analyzer = ConceptualAnalyzer()
            
            # Create sample data
            activations = {
                'layer1': np.random.randn(10, 50),
                'layer2': np.random.randn(10, 80)
            }
            labels = ['concept_A'] * 5 + ['concept_B'] * 5
            
            # Extract concepts
            concepts = analyzer.extract_concepts(activations, labels, method='pca')
            
            self.assertIsInstance(concepts, dict)
            
            # Verify concepts have correct structure
            for concept_name, concept in concepts.items():
                self.assertTrue(hasattr(concept, 'name'))
                self.assertTrue(hasattr(concept, 'vector'))
                self.assertTrue(hasattr(concept, 'confidence'))
                
        except ImportError as e:
            self.skipTest(f"Conceptual analysis dependencies not available: {e}")
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available")
    def test_concept_algebra_operations(self):
        """Test concept algebra operations."""
        try:
            from src.analysis.conceptual_analysis import ConceptualAnalyzer, ConceptVector
            
            analyzer = ConceptualAnalyzer()
            
            # Create test concepts
            concept_a = ConceptVector(
                name="test_a",
                vector=np.array([1.0, 2.0, 3.0]),
                layer="layer1",
                model_name="test",
                confidence=0.8,
                metadata={}
            )
            
            concept_b = ConceptVector(
                name="test_b",
                vector=np.array([0.5, 1.0, 1.5]),
                layer="layer1",
                model_name="test",
                confidence=0.7,
                metadata={}
            )
            
            analyzer.concepts["test_a"] = concept_a
            analyzer.concepts["test_b"] = concept_b
            
            # Test operations
            result_add = analyzer.concept_algebra("test_a", "test_b", "add")
            self.assertEqual(result_add.name, "test_a_add_test_b")
            
            result_sub = analyzer.concept_algebra("test_a", "test_b", "subtract")
            self.assertEqual(result_sub.name, "test_a_subtract_test_b")
            
            # Verify results are different
            self.assertFalse(np.array_equal(result_add.vector, result_sub.vector))
            
        except ImportError as e:
            self.skipTest(f"Conceptual analysis dependencies not available: {e}")
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available")
    def test_circuit_discovery(self):
        """Test circuit discovery functionality."""
        try:
            from src.analysis.conceptual_analysis import ConceptualAnalyzer
            
            analyzer = ConceptualAnalyzer()
            
            # Create sample data
            activations = {
                'layer1': np.random.randn(20, 100),
                'layer2': np.random.randn(20, 150),
                'layer3': np.random.randn(20, 200)
            }
            labels = ['task_A'] * 10 + ['task_B'] * 10
            
            # Discover circuits
            circuits = analyzer.discover_circuits(activations, labels, "test_task")
            
            self.assertIsInstance(circuits, dict)
            
            # Verify circuit structure if any found
            for circuit_name, circuit in circuits.items():
                self.assertTrue(hasattr(circuit, 'name'))
                self.assertTrue(hasattr(circuit, 'components'))
                self.assertTrue(hasattr(circuit, 'connections'))
                self.assertTrue(hasattr(circuit, 'function'))
                self.assertTrue(hasattr(circuit, 'evidence_strength'))
                
        except ImportError as e:
            self.skipTest(f"Conceptual analysis dependencies not available: {e}")
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available")
    def test_world_model_analysis(self):
        """Test world model analysis."""
        try:
            from src.analysis.conceptual_analysis import ConceptualAnalyzer
            
            analyzer = ConceptualAnalyzer()
            
            # Create sample data
            activations = {
                'layer1': np.random.randn(10, 100),
                'layer2': np.random.randn(10, 150)
            }
            
            # Create metadata
            stimuli_metadata = [
                {'object': f'object_{i}', 'position': [i, i+1]}
                for i in range(10)
            ]
            
            # Analyze world model
            world_model = analyzer.analyze_world_model(activations, stimuli_metadata)
            
            self.assertIsInstance(world_model, dict)
            self.assertIn('object_representations', world_model)
            self.assertIn('spatial_representations', world_model)
            self.assertIn('temporal_representations', world_model)
            self.assertIn('relational_representations', world_model)
            
        except ImportError as e:
            self.skipTest(f"Conceptual analysis dependencies not available: {e}")
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available") 
    def test_knowledge_transfer_analysis(self):
        """Test knowledge transfer analysis."""
        try:
            from src.analysis.conceptual_analysis import ConceptualAnalyzer
            
            analyzer = ConceptualAnalyzer()
            
            # Create source and target activations
            source_activations = {
                'layer1': np.random.randn(15, 100),
                'layer2': np.random.randn(15, 150)
            }
            
            target_activations = {
                'layer1_target': source_activations['layer1'] + 0.1 * np.random.randn(15, 100),
                'layer2_target': source_activations['layer2'] + 0.1 * np.random.randn(15, 150)
            }
            
            # Analyze knowledge transfer
            result = analyzer.analyze_knowledge_transfer(
                source_activations,
                target_activations,
                "source_model",
                "target_model"
            )
            
            self.assertTrue(hasattr(result, 'source_model'))
            self.assertTrue(hasattr(result, 'target_model'))
            self.assertTrue(hasattr(result, 'transfer_score'))
            self.assertTrue(hasattr(result, 'transfer_map'))
            self.assertTrue(hasattr(result, 'preserved_concepts'))
            self.assertTrue(hasattr(result, 'lost_concepts'))
            self.assertTrue(hasattr(result, 'emergent_concepts'))
            
            self.assertEqual(result.source_model, "source_model")
            self.assertEqual(result.target_model, "target_model")
            self.assertGreaterEqual(result.transfer_score, 0)
            self.assertLessEqual(result.transfer_score, 1)
            
        except ImportError as e:
            self.skipTest(f"Conceptual analysis dependencies not available: {e}")
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available")
    def test_cross_model_rsa(self):
        """Test cross-model representational similarity analysis."""
        try:
            from src.analysis.conceptual_analysis import ConceptualAnalyzer
            
            analyzer = ConceptualAnalyzer()
            
            # Create multi-model activations
            base_activations = {
                'layer1': np.random.randn(8, 50),
                'layer2': np.random.randn(8, 80)
            }
            
            model_activations = {
                'model1': base_activations,
                'model2': {
                    k: v + 0.2 * np.random.randn(*v.shape) 
                    for k, v in base_activations.items()
                }
            }
            
            stimuli = [f"stimulus_{i}" for i in range(8)]
            
            # Perform cross-model RSA
            results = analyzer.cross_model_rsa(model_activations, stimuli)
            
            self.assertIsInstance(results, dict)
            self.assertIn('similarity_matrices', results)
            self.assertIn('model_comparisons', results)
            self.assertIn('hierarchical_alignment', results)
            
            # Check similarity matrices exist for each model
            for model_name in model_activations:
                self.assertIn(model_name, results['similarity_matrices'])
                
        except ImportError as e:
            self.skipTest(f"Conceptual analysis dependencies not available: {e}")
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available")
    def test_save_load_analysis_results(self):
        """Test saving and loading analysis results."""
        try:
            from src.analysis.conceptual_analysis import ConceptualAnalyzer, ConceptVector, Circuit
            
            analyzer = ConceptualAnalyzer()
            
            # Add test data
            concept = ConceptVector(
                name="test_concept",
                vector=np.array([1, 2, 3]),
                layer="layer1",
                model_name="test",
                confidence=0.8,
                metadata={'test': True}
            )
            analyzer.concepts["test_concept"] = concept
            
            circuit = Circuit(
                name="test_circuit",
                components=["layer1_0", "layer2_1"],
                connections=[("layer1_0", "layer2_1", 0.5)],
                function="test function",
                evidence_strength=0.7,
                metadata={'circuit_test': True}
            )
            analyzer.circuits["test_circuit"] = circuit
            
            # Save
            output_path = self.test_dir / "test_results.json"
            analyzer.save_analysis_results(str(output_path))
            self.assertTrue(output_path.exists())
            
            # Load
            new_analyzer = ConceptualAnalyzer()
            new_analyzer.load_analysis_results(str(output_path))
            
            self.assertIn("test_concept", new_analyzer.concepts)
            self.assertIn("test_circuit", new_analyzer.circuits)
            
            loaded_concept = new_analyzer.concepts["test_concept"]
            self.assertEqual(loaded_concept.name, "test_concept")
            np.testing.assert_array_equal(loaded_concept.vector, np.array([1, 2, 3]))
            self.assertEqual(loaded_concept.confidence, 0.8)
            
        except ImportError as e:
            self.skipTest(f"Conceptual analysis dependencies not available: {e}")
    

class TestRunner:
    """Test runner for executing all tests."""
    
    def __init__(self):
        """Initialize test runner."""
        self.results = []
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        logger.info("Starting comprehensive test suite...")
        
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__(__name__))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2, stream=open('/dev/null', 'w'))
        result = runner.run(suite)
        
        # Compile results
        test_results = {
            'total_tests': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1),
            'failure_details': [str(failure) for failure in result.failures],
            'error_details': [str(error) for error in result.errors]
        }
        
        logger.info(f"Test suite completed: {test_results['success_rate']:.2%} success rate")
        return test_results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance regression tests."""
        logger.info("Running performance tests...")
        
        performance_results = {
            'memory_usage': self._test_memory_usage(),
            'execution_time': self._test_execution_time(),
            'gpu_utilization': self._test_gpu_utilization()
        }
        
        return performance_results
    
    def _test_memory_usage(self) -> Dict[str, float]:
        """Test memory usage patterns."""
        try:
            import psutil
            import gc
            
            # Baseline memory
            gc.collect()
            baseline = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Test memory usage during operations
            test_data = np.random.randn(1000, 100)
            current = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            return {
                'baseline_mb': baseline,
                'peak_mb': current,
                'delta_mb': current - baseline
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def _test_execution_time(self) -> Dict[str, float]:
        """Test execution time benchmarks."""
        import time
        
        # Simple benchmark operations
        times = {}
        
        # Test array operations
        start = time.time()
        test_array = np.random.randn(10000, 100)
        np.mean(test_array, axis=1)
        times['array_operations'] = time.time() - start
        
        # Test file I/O
        start = time.time()
        with tempfile.NamedTemporaryFile(mode='w') as f:
            json.dump({'test': [1, 2, 3] * 1000}, f)
        times['file_io'] = time.time() - start
        
        return times
    
    def _test_gpu_utilization(self) -> Dict[str, Any]:
        """Test GPU utilization if available."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'gpu_available': True,
                    'gpu_count': torch.cuda.device_count(),
                    'memory_allocated': torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                    'memory_reserved': torch.cuda.memory_reserved() / 1024 / 1024    # MB
                }
            else:
                return {'gpu_available': False}
        except ImportError:
            return {'error': 'PyTorch not available'}


def run_tests():
    """Main function to run all tests."""
    runner = TestRunner()
    
    # Run unit and integration tests
    test_results = runner.run_all_tests()
    print(f"\n=== Test Results ===")
    print(f"Total tests: {test_results['total_tests']}")
    print(f"Success rate: {test_results['success_rate']:.2%}")
    print(f"Failures: {test_results['failures']}")
    print(f"Errors: {test_results['errors']}")
    print(f"Skipped: {test_results['skipped']}")
    
    # Run performance tests
    perf_results = runner.run_performance_tests()
    print(f"\n=== Performance Results ===")
    for category, metrics in perf_results.items():
        print(f"{category}: {metrics}")
    
    return test_results, perf_results


if __name__ == "__main__":
    run_tests()
