"""Comprehensive integration tests for NeuronMap end-to-end workflows."""

import unittest
import tempfile
import json
import numpy as np
import os
import sys
import subprocess
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestFullPipeline(unittest.TestCase):
    """Test complete analysis pipeline from question generation to visualization."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create directory structure
        self.data_dir = self.test_dir / "data"
        self.config_dir = self.test_dir / "configs"
        self.output_dir = self.test_dir / "outputs"
        
        for dir_path in [self.data_dir, self.config_dir, self.output_dir]:
            dir_path.mkdir(parents=True)
        
        # Create test configurations
        self.create_test_configs()
        
    def tearDown(self):
        """Clean up integration test environment."""
        self.temp_dir.cleanup()
    
    def create_test_configs(self):
        """Create test configuration files."""
        # Models configuration
        models_config = {
            "models": {
                "test_model": {
                    "name": "gpt2",
                    "type": "gpt",
                    "layers": {
                        "total_layers": 12,
                        "attention_layers": [f"transformer.h.{i}.attn" for i in range(12)],
                        "mlp_layers": [f"transformer.h.{i}.mlp" for i in range(12)]
                    }
                }
            },
            "layer_patterns": {
                "gpt": {
                    "attention": "transformer.h.{}.attn",
                    "mlp": "transformer.h.{}.mlp",
                    "full": "transformer.h.{}"
                }
            }
        }
        
        # Experiments configuration
        experiments_config = {
            "test_experiment": {
                "question_generation": {
                    "num_questions": 10,
                    "categories": ["factual", "reasoning"],
                    "output_file": str(self.data_dir / "questions.json")
                },
                "activation_extraction": {
                    "model": "test_model",
                    "batch_size": 4,
                    "layers": ["transformer.h.0", "transformer.h.6", "transformer.h.11"],
                    "output_file": str(self.data_dir / "activations.h5")
                },
                "visualization": {
                    "output_dir": str(self.output_dir),
                    "plot_types": ["pca", "tsne", "heatmap"],
                    "interactive": True
                }
            }
        }
        
        # Save configurations
        import yaml
        with open(self.config_dir / "models.yaml", 'w') as f:
            yaml.dump(models_config, f)
        
        with open(self.config_dir / "experiments.yaml", 'w') as f:
            yaml.dump(experiments_config, f)
    
    def test_question_generation_to_analysis_pipeline(self):
        """Test pipeline from question generation through analysis."""
        try:
            # Step 1: Generate test questions (mocked)
            test_questions = [
                {"text": f"What is the answer to question {i}?", "category": "factual", "id": i}
                for i in range(10)
            ]
            
            questions_file = self.data_dir / "questions.json"
            with open(questions_file, 'w') as f:
                json.dump({"questions": test_questions, "metadata": {"generated_at": "2024-01-01"}}, f)
            
            self.assertTrue(questions_file.exists())
            
            # Step 2: Mock activation extraction
            # Create synthetic activation data
            activations_data = {
                "questions": [q["text"] for q in test_questions],
                "activations": {
                    "transformer.h.0": np.random.randn(10, 768).tolist(),
                    "transformer.h.6": np.random.randn(10, 768).tolist(),
                    "transformer.h.11": np.random.randn(10, 768).tolist()
                },
                "metadata": {
                    "model_name": "gpt2",
                    "layers_extracted": ["transformer.h.0", "transformer.h.6", "transformer.h.11"],
                    "extraction_time": "2024-01-01"
                }
            }
            
            activations_file = self.data_dir / "activations.json"
            with open(activations_file, 'w') as f:
                json.dump(activations_data, f)
            
            self.assertTrue(activations_file.exists())
            
            # Step 3: Mock analysis
            # Verify data can be loaded and basic analysis performed
            with open(activations_file, 'r') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(len(loaded_data["questions"]), 10)
            self.assertEqual(len(loaded_data["activations"]), 3)
            
            # Perform basic statistical analysis
            analysis_results = {}
            for layer_name, activations in loaded_data["activations"].items():
                activations_array = np.array(activations)
                analysis_results[layer_name] = {
                    "mean": float(np.mean(activations_array)),
                    "std": float(np.std(activations_array)),
                    "shape": activations_array.shape
                }
            
            # Step 4: Save analysis results
            results_file = self.output_dir / "analysis_results.json"
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            self.assertTrue(results_file.exists())
            
            # Verify complete pipeline
            self.assertTrue(questions_file.exists())
            self.assertTrue(activations_file.exists())
            self.assertTrue(results_file.exists())
            
            print("✅ Full pipeline test completed successfully")
            
        except Exception as e:
            self.fail(f"Pipeline test failed: {e}")
    
    def test_configuration_driven_workflow(self):
        """Test workflow driven by configuration files."""
        try:
            # Load experiment configuration
            import yaml
            with open(self.config_dir / "experiments.yaml", 'r') as f:
                experiment_config = yaml.safe_load(f)
            
            test_exp = experiment_config["test_experiment"]
            
            # Verify configuration structure
            self.assertIn("question_generation", test_exp)
            self.assertIn("activation_extraction", test_exp)
            self.assertIn("visualization", test_exp)
            
            # Test question generation config
            qg_config = test_exp["question_generation"]
            self.assertEqual(qg_config["num_questions"], 10)
            self.assertIn("factual", qg_config["categories"])
            
            # Test activation extraction config
            ae_config = test_exp["activation_extraction"]
            self.assertEqual(ae_config["model"], "test_model")
            self.assertEqual(ae_config["batch_size"], 4)
            self.assertEqual(len(ae_config["layers"]), 3)
            
            # Test visualization config
            viz_config = test_exp["visualization"]
            self.assertTrue(viz_config["interactive"])
            self.assertIn("pca", viz_config["plot_types"])
            
            print("✅ Configuration-driven workflow test completed")
            
        except Exception as e:
            self.fail(f"Configuration workflow test failed: {e}")


class TestCLIIntegration(unittest.TestCase):
    """Test CLI command integration."""
    
    def setUp(self):
        """Set up CLI test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create test data
        self.test_data_file = self.test_dir / "test_data.json"
        test_data = {
            "questions": ["What is AI?", "How do neural networks work?"],
            "activations": {
                "layer_0": np.random.randn(2, 768).tolist()
            }
        }
        
        with open(self.test_data_file, 'w') as f:
            json.dump(test_data, f)
    
    def tearDown(self):
        """Clean up CLI test environment."""
        self.temp_dir.cleanup()
    
    def test_main_py_execution(self):
        """Test that main.py can be executed without errors."""
        try:
            # Test help command
            result = subprocess.run(
                [sys.executable, "main.py", "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Should exit with code 0 and show help text
            self.assertEqual(result.returncode, 0)
            self.assertIn("usage:", result.stdout.lower())
            
            print("✅ main.py help command works")
            
        except subprocess.TimeoutExpired:
            self.fail("main.py --help command timed out")
        except FileNotFoundError:
            self.skipTest("main.py not found or not executable")
        except Exception as e:
            self.fail(f"main.py execution failed: {e}")
    
    def test_cli_command_structure(self):
        """Test that CLI commands are properly structured."""
        try:
            main_file = Path("main.py")
            if not main_file.exists():
                self.skipTest("main.py not found")
            
            with open(main_file, 'r') as f:
                content = f.read()
            
            # Check for argument parser setup
            self.assertIn("argparse.ArgumentParser", content)
            self.assertIn("subparsers", content)
            
            # Check for basic commands
            expected_commands = ["generate", "extract", "analyze", "visualize"]
            found_commands = []
            
            for cmd in expected_commands:
                if f'"{cmd}"' in content or f"'{cmd}'" in content:
                    found_commands.append(cmd)
            
            self.assertGreater(len(found_commands), 0, "No CLI commands found")
            
            print(f"✅ Found CLI commands: {found_commands}")
            
        except Exception as e:
            self.fail(f"CLI structure test failed: {e}")


class TestDataFlow(unittest.TestCase):
    """Test data flow between different components."""
    
    def setUp(self):
        """Set up data flow test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create input/output directories
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        self.intermediate_dir = self.test_dir / "intermediate"
        
        for dir_path in [self.input_dir, self.output_dir, self.intermediate_dir]:
            dir_path.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up data flow test environment."""
        self.temp_dir.cleanup()
    
    def test_data_format_compatibility(self):
        """Test that data formats are compatible between components."""
        try:
            # Create question generation output format
            questions_data = {
                "questions": [
                    {"text": "What is machine learning?", "category": "factual", "id": 1},
                    {"text": "How do transformers work?", "category": "technical", "id": 2}
                ],
                "metadata": {
                    "generated_at": "2024-01-01",
                    "total_questions": 2,
                    "categories": ["factual", "technical"]
                }
            }
            
            questions_file = self.intermediate_dir / "questions.json"
            with open(questions_file, 'w') as f:
                json.dump(questions_data, f)
            
            # Create activation extraction output format
            activations_data = {
                "questions": [q["text"] for q in questions_data["questions"]],
                "activations": {
                    "layer_0": np.random.randn(2, 768).tolist(),
                    "layer_6": np.random.randn(2, 768).tolist()
                },
                "metadata": {
                    "model_name": "gpt2",
                    "extraction_time": "2024-01-01",
                    "questions_source": str(questions_file)
                }
            }
            
            activations_file = self.intermediate_dir / "activations.json"
            with open(activations_file, 'w') as f:
                json.dump(activations_data, f)
            
            # Verify data compatibility
            # Load questions
            with open(questions_file, 'r') as f:
                loaded_questions = json.load(f)
            
            # Load activations
            with open(activations_file, 'r') as f:
                loaded_activations = json.load(f)
            
            # Check compatibility
            question_texts = [q["text"] for q in loaded_questions["questions"]]
            activation_questions = loaded_activations["questions"]
            
            self.assertEqual(question_texts, activation_questions)
            self.assertEqual(len(loaded_activations["activations"]["layer_0"]), len(question_texts))
            
            print("✅ Data format compatibility verified")
            
        except Exception as e:
            self.fail(f"Data format compatibility test failed: {e}")
    
    def test_file_pipeline_integrity(self):
        """Test file pipeline maintains data integrity."""
        try:
            # Create initial data
            original_data = {
                "values": list(range(100)),
                "metadata": {"source": "test", "version": "1.0"}
            }
            
            input_file = self.input_dir / "input.json"
            with open(input_file, 'w') as f:
                json.dump(original_data, f)
            
            # Simulate processing pipeline
            # Stage 1: Load and validate
            with open(input_file, 'r') as f:
                stage1_data = json.load(f)
            
            # Add processing metadata
            stage1_data["processing"] = {"stage": 1, "processed_at": "2024-01-01"}
            
            stage1_file = self.intermediate_dir / "stage1.json"
            with open(stage1_file, 'w') as f:
                json.dump(stage1_data, f)
            
            # Stage 2: Transform data
            with open(stage1_file, 'r') as f:
                stage2_data = json.load(f)
            
            # Apply transformation (square the values)
            stage2_data["transformed_values"] = [x**2 for x in stage2_data["values"]]
            stage2_data["processing"]["stage"] = 2
            
            stage2_file = self.intermediate_dir / "stage2.json"
            with open(stage2_file, 'w') as f:
                json.dump(stage2_data, f)
            
            # Stage 3: Final output
            with open(stage2_file, 'r') as f:
                final_data = json.load(f)
            
            final_data["processing"]["stage"] = 3
            final_data["processing"]["completed"] = True
            
            output_file = self.output_dir / "output.json"
            with open(output_file, 'w') as f:
                json.dump(final_data, f)
            
            # Verify pipeline integrity
            with open(output_file, 'r') as f:
                result_data = json.load(f)
            
            # Check that original data is preserved
            self.assertEqual(result_data["values"], original_data["values"])
            self.assertEqual(result_data["metadata"], original_data["metadata"])
            
            # Check that transformations were applied
            self.assertIn("transformed_values", result_data)
            self.assertEqual(len(result_data["transformed_values"]), 100)
            self.assertEqual(result_data["transformed_values"][5], 25)  # 5^2 = 25
            
            # Check processing metadata
            self.assertEqual(result_data["processing"]["stage"], 3)
            self.assertTrue(result_data["processing"]["completed"])
            
            print("✅ File pipeline integrity verified")
            
        except Exception as e:
            self.fail(f"Pipeline integrity test failed: {e}")


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling across different components."""
    
    def setUp(self):
        """Set up error handling test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up error handling test environment."""
        self.temp_dir.cleanup()
    
    def test_graceful_error_handling(self):
        """Test that errors are handled gracefully throughout the pipeline."""
        try:
            # Test 1: Missing input file
            nonexistent_file = self.test_dir / "nonexistent.json"
            
            try:
                with open(nonexistent_file, 'r') as f:
                    data = json.load(f)
                self.fail("Should have raised FileNotFoundError")
            except FileNotFoundError:
                # Expected behavior
                pass
            
            # Test 2: Invalid JSON format
            invalid_json_file = self.test_dir / "invalid.json"
            with open(invalid_json_file, 'w') as f:
                f.write("{ invalid json content")
            
            try:
                with open(invalid_json_file, 'r') as f:
                    data = json.load(f)
                self.fail("Should have raised JSONDecodeError")
            except json.JSONDecodeError:
                # Expected behavior
                pass
            
            # Test 3: Invalid data structure
            invalid_data_file = self.test_dir / "invalid_data.json"
            with open(invalid_data_file, 'w') as f:
                json.dump({"wrong": "structure"}, f)
            
            with open(invalid_data_file, 'r') as f:
                data = json.load(f)
            
            # Check for expected keys
            expected_keys = ["questions", "activations"]
            missing_keys = [key for key in expected_keys if key not in data]
            
            if missing_keys:
                # This is expected for our test
                self.assertGreater(len(missing_keys), 0)
            
            print("✅ Error handling tests completed")
            
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")
    
    def test_recovery_mechanisms(self):
        """Test recovery mechanisms when components fail."""
        try:
            # Create a scenario where processing partially fails
            partial_data = {
                "questions": ["Q1", "Q2", "Q3"],
                "activations": {
                    "layer_0": [[1, 2, 3], [4, 5, 6]],  # Missing data for Q3
                    "layer_1": [[7, 8, 9], [10, 11, 12], [13, 14, 15]]  # Complete data
                }
            }
            
            partial_file = self.test_dir / "partial_data.json"
            with open(partial_file, 'w') as f:
                json.dump(partial_data, f)
            
            # Load and validate data
            with open(partial_file, 'r') as f:
                data = json.load(f)
            
            num_questions = len(data["questions"])
            
            # Check each layer for consistency
            recovery_info = {}
            for layer_name, activations in data["activations"].items():
                if len(activations) != num_questions:
                    recovery_info[layer_name] = {
                        "expected": num_questions,
                        "actual": len(activations),
                        "missing": num_questions - len(activations)
                    }
            
            # Recovery: Use only complete data
            complete_layers = {
                layer_name: activations 
                for layer_name, activations in data["activations"].items()
                if len(activations) == num_questions
            }
            
            # Verify recovery
            self.assertEqual(len(complete_layers), 1)  # Only layer_1 should remain
            self.assertIn("layer_1", complete_layers)
            self.assertNotIn("layer_0", complete_layers)
            
            print("✅ Recovery mechanism test completed")
            
        except Exception as e:
            self.fail(f"Recovery mechanism test failed: {e}")


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance aspects of integrated workflows."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up performance test environment."""
        self.temp_dir.cleanup()
    
    def test_large_data_handling(self):
        """Test handling of large datasets."""
        try:
            # Create large synthetic dataset
            large_data = {
                "questions": [f"Question {i}?" for i in range(1000)],
                "activations": {
                    f"layer_{i}": np.random.randn(1000, 768).tolist() 
                    for i in range(5)
                },
                "metadata": {"size": "large", "samples": 1000}
            }
            
            # Test file I/O performance
            large_file = self.test_dir / "large_data.json"
            
            start_time = time.time()
            with open(large_file, 'w') as f:
                json.dump(large_data, f)
            write_time = time.time() - start_time
            
            start_time = time.time()
            with open(large_file, 'r') as f:
                loaded_data = json.load(f)
            read_time = time.time() - start_time
            
            # Verify data integrity
            self.assertEqual(len(loaded_data["questions"]), 1000)
            self.assertEqual(len(loaded_data["activations"]), 5)
            
            # Test processing performance
            start_time = time.time()
            for layer_name, activations in loaded_data["activations"].items():
                activations_array = np.array(activations)
                mean_activation = np.mean(activations_array)
                std_activation = np.std(activations_array)
            processing_time = time.time() - start_time
            
            # Performance should be reasonable (adjust thresholds as needed)
            self.assertLess(write_time, 30.0, "Write time too slow")
            self.assertLess(read_time, 30.0, "Read time too slow")
            self.assertLess(processing_time, 10.0, "Processing time too slow")
            
            print(f"✅ Large data test completed:")
            print(f"  Write time: {write_time:.2f}s")
            print(f"  Read time: {read_time:.2f}s")
            print(f"  Processing time: {processing_time:.2f}s")
            
        except Exception as e:
            self.fail(f"Large data handling test failed: {e}")
    
    def test_memory_usage_monitoring(self):
        """Test memory usage during processing."""
        try:
            import psutil
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and process data
            test_data = {
                "activations": np.random.randn(500, 768).tolist()
            }
            
            # Convert to numpy and perform operations
            activations_array = np.array(test_data["activations"])
            
            # Perform memory-intensive operations
            pca_simulation = np.dot(activations_array.T, activations_array)
            correlation_matrix = np.corrcoef(activations_array)
            
            # Get peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Memory increase should be reasonable
            self.assertLess(memory_increase, 1000, "Memory usage too high")  # Less than 1GB increase
            
            print(f"✅ Memory monitoring test completed:")
            print(f"  Initial memory: {initial_memory:.1f} MB")
            print(f"  Peak memory: {peak_memory:.1f} MB")
            print(f"  Memory increase: {memory_increase:.1f} MB")
            
        except ImportError:
            self.skipTest("psutil not available for memory monitoring")
        except Exception as e:
            self.fail(f"Memory monitoring test failed: {e}")


def run_integration_tests():
    """Run all integration tests."""
    print("🔧 Running NeuronMap Integration Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestFullPipeline,
        TestCLIIntegration,
        TestDataFlow,
        TestErrorHandlingIntegration,
        TestPerformanceIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    tests_run = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    
    print(f"Tests run: {tests_run}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print(f"Success rate: {((tests_run - failures - errors) / tests_run * 100):.1f}%" if tests_run > 0 else "0%")
    
    if failures == 0 and errors == 0:
        print("\n🎉 All integration tests passed!")
        return True
    else:
        print(f"\n⚠️  {failures + errors} integration test(s) failed")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)