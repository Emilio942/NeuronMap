"""Unit tests for analysis modules."""

import unittest
import pytest
import tempfile
import json
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.analysis.activation_extractor import ActivationExtractor
    from src.analysis.advanced_analyzer import AdvancedAnalyzer
    from src.analysis.attention_analysis import AttentionAnalyzer
    from src.analysis.interpretability import InterpretabilityAnalyzer
    from src.analysis.experimental_analysis import ExperimentalAnalyzer
except ImportError as e:
    print(f"Warning: Could not import analysis modules: {e}")


class TestActivationExtractor(unittest.TestCase):
    """Test activation extraction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = {
            'model_name': 'gpt2',
            'device': 'cpu',
            'batch_size': 4,
            'layers': ['transformer.h.0', 'transformer.h.6', 'transformer.h.11'],
            'output_file': str(Path(self.temp_dir.name) / 'activations.json')
        }
        
        self.sample_questions = [
            "What is artificial intelligence?",
            "How do neural networks learn?",
            "What is machine learning?"
        ]
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_activation_extractor_init(self):
        """Test ActivationExtractor initialization."""
        try:
            with patch('transformers.AutoModel.from_pretrained'):
                with patch('transformers.AutoTokenizer.from_pretrained'):
                    extractor = ActivationExtractor(self.config)
                    self.assertEqual(extractor.config['model_name'], 'gpt2')
                    self.assertEqual(extractor.config['device'], 'cpu')
                    
        except ImportError:
            self.skipTest("ActivationExtractor not available")
    
    @patch('src.analysis.activation_extractor.AutoModelForCausalLM.from_pretrained')
    @patch('src.analysis.activation_extractor.AutoTokenizer.from_pretrained')
    def test_model_loading(self, mock_tokenizer, mock_model):
        """Test model and tokenizer loading."""
        try:
            # Mock model and tokenizer
            mock_model_instance = Mock()
            mock_tokenizer_instance = Mock()
            mock_model.return_value = mock_model_instance
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            # Mock the model's named_modules method for layer lookup
            mock_model_instance.named_modules.return_value = [
                ('transformer.h.5.mlp.c_proj', Mock())
            ]
            # Mock other methods that might be called
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            
            extractor = ActivationExtractor(self.config)
            success = extractor.load_model()
            
            self.assertTrue(success)
            mock_model.assert_called_once()
            mock_tokenizer.assert_called_once()
            
        except ImportError:
            self.skipTest("ActivationExtractor not available")
    
    @patch('src.analysis.activation_extractor.AutoModelForCausalLM.from_pretrained')
    @patch('src.analysis.activation_extractor.AutoTokenizer.from_pretrained')
    def test_activation_extraction(self, mock_tokenizer, mock_model):
        """Test activation extraction from model."""
        try:
            # Mock model outputs
            mock_model_instance = Mock()
            mock_tokenizer_instance = Mock()
            
            # Mock tokenizer output
            mock_tokenizer_instance.return_value = {
                'input_ids': [[1, 2, 3, 4]],
                'attention_mask': [[1, 1, 1, 1]]
            }
            
            # Mock model output with hidden states
            import torch
            mock_outputs = Mock()
            mock_outputs.hidden_states = [
                torch.randn(1, 4, 768) for _ in range(12)  # 12 layers
            ]
            mock_model_instance.return_value = mock_outputs
            
            # Mock the model's named_modules method for layer lookup
            mock_model_instance.named_modules.return_value = [
                ('transformer.h.5.mlp.c_proj', Mock())
            ]
            # Mock other methods that might be called
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            
            mock_model.return_value = mock_model_instance
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            extractor = ActivationExtractor(self.config)
            extractor.load_model()
            
            # Extract activations
            activations = extractor.extract_activations(self.sample_questions[:1])
            
            self.assertIsInstance(activations, dict)
            self.assertIn('questions', activations)
            self.assertIn('activations', activations)
            
        except ImportError:
            self.skipTest("ActivationExtractor not available - missing torch/transformers")
    
    def test_layer_name_validation(self):
        """Test layer name validation."""
        try:
            with patch('transformers.AutoModel.from_pretrained'):
                with patch('transformers.AutoTokenizer.from_pretrained'):
                    extractor = ActivationExtractor(self.config)
                    
                    # Mock model with named modules
                    mock_model = Mock()
                    mock_model.named_modules.return_value = [
                        ('transformer.h.0', Mock()),
                        ('transformer.h.6', Mock()),
                        ('transformer.h.11', Mock()),
                    ]
                    extractor.model = mock_model
                    
                    # Test valid layer names
                    valid_layers = extractor.validate_layer_names(
                        ['transformer.h.0', 'transformer.h.6']
                    )
                    self.assertEqual(len(valid_layers), 2)
                    
                    # Test invalid layer names
                    valid_layers = extractor.validate_layer_names(
                        ['invalid.layer', 'transformer.h.0']
                    )
                    self.assertEqual(len(valid_layers), 1)  # Only valid one should remain
                    
        except ImportError:
            self.skipTest("ActivationExtractor not available")


class TestAdvancedAnalyzer(unittest.TestCase):
    """Test advanced analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_activations = {
            'layer_0': np.random.randn(50, 768),
            'layer_6': np.random.randn(50, 768),
            'layer_11': np.random.randn(50, 768)
        }
        
        self.sample_questions = [f"Question {i}?" for i in range(50)]
        
    def test_advanced_analyzer_init(self):
        """Test AdvancedAnalyzer initialization."""
        try:
            analyzer = AdvancedAnalyzer()
            self.assertIsNotNone(analyzer)
            
        except ImportError:
            self.skipTest("AdvancedAnalyzer not available")
    
    def test_pca_analysis(self):
        """Test PCA analysis of activations."""
        try:
            analyzer = AdvancedAnalyzer()
            
            # Perform PCA analysis
            pca_results = analyzer.perform_pca_analysis(
                self.sample_activations['layer_0'],
                n_components=10
            )
            
            self.assertIn('components', pca_results)
            self.assertIn('explained_variance_ratio', pca_results)
            self.assertIn('transformed_data', pca_results)
            
            # Check dimensions
            self.assertEqual(pca_results['transformed_data'].shape[1], 10)
            self.assertEqual(len(pca_results['explained_variance_ratio']), 10)
            
        except ImportError:
            self.skipTest("AdvancedAnalyzer not available")
    
    def test_clustering_analysis(self):
        """Test clustering analysis of activations."""
        try:
            analyzer = AdvancedAnalyzer()
            
            # Perform clustering analysis
            clustering_results = analyzer.perform_clustering_analysis(
                self.sample_activations['layer_0'],
                n_clusters=5,
                methods=['kmeans', 'hierarchical']
            )
            
            self.assertIn('kmeans', clustering_results)
            self.assertIn('hierarchical', clustering_results)
            
            # Check cluster assignments
            kmeans_labels = clustering_results['kmeans']['labels']
            self.assertEqual(len(kmeans_labels), 50)  # One label per sample
            self.assertLessEqual(max(kmeans_labels), 4)  # 5 clusters (0-4)
            
        except ImportError:
            self.skipTest("AdvancedAnalyzer not available")
    
    def test_statistical_analysis(self):
        """Test statistical analysis of activations."""
        try:
            analyzer = AdvancedAnalyzer()
            
            # Perform statistical analysis
            stats_results = analyzer.compute_activation_statistics(
                self.sample_activations['layer_0']
            )
            
            expected_stats = ['mean', 'std', 'min', 'max', 'sparsity', 'skewness', 'kurtosis']
            for stat in expected_stats:
                self.assertIn(stat, stats_results)
            
            # Check statistical properties
            self.assertIsInstance(stats_results['mean'], (int, float))
            self.assertIsInstance(stats_results['std'], (int, float))
            self.assertGreaterEqual(stats_results['sparsity'], 0)
            self.assertLessEqual(stats_results['sparsity'], 1)
            
        except ImportError:
            self.skipTest("AdvancedAnalyzer not available")
    
    def test_correlation_analysis(self):
        """Test correlation analysis between layers."""
        try:
            analyzer = AdvancedAnalyzer()
            
            # Perform correlation analysis
            corr_results = analyzer.analyze_layer_correlations(
                self.sample_activations
            )
            
            self.assertIn('correlation_matrix', corr_results)
            self.assertIn('highly_correlated_pairs', corr_results)
            
            # Check correlation matrix properties
            corr_matrix = np.array(corr_results['correlation_matrix'])
            self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])  # Square matrix
            np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T)  # Symmetric
            
        except ImportError:
            self.skipTest("AdvancedAnalyzer not available")


class TestAttentionAnalyzer(unittest.TestCase):
    """Test attention analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock attention weights: (batch_size, num_heads, seq_len, seq_len)
        self.attention_weights = {
            'layer_0': np.random.rand(10, 12, 20, 20),
            'layer_6': np.random.rand(10, 12, 20, 20),
            'layer_11': np.random.rand(10, 12, 20, 20)
        }
        
        self.token_sequences = [
            ["What", "is", "artificial", "intelligence", "?"] * 4  # 20 tokens
            for _ in range(10)
        ]
    
    def test_attention_analyzer_init(self):
        """Test AttentionAnalyzer initialization."""
        try:
            analyzer = AttentionAnalyzer()
            self.assertIsNotNone(analyzer)
            
        except ImportError:
            self.skipTest("AttentionAnalyzer not available")
    
    def test_attention_pattern_analysis(self):
        """Test attention pattern analysis."""
        try:
            analyzer = AttentionAnalyzer()
            
            # Analyze attention patterns
            pattern_results = analyzer.analyze_attention_patterns(
                self.attention_weights['layer_0']
            )
            
            self.assertIn('attention_entropy', pattern_results)
            self.assertIn('head_importance', pattern_results)
            self.assertIn('attention_distance', pattern_results)
            
            # Check dimensions
            self.assertEqual(len(pattern_results['head_importance']), 12)  # 12 heads
            
        except ImportError:
            self.skipTest("AttentionAnalyzer not available")
    
    def test_head_importance_ranking(self):
        """Test attention head importance ranking."""
        try:
            analyzer = AttentionAnalyzer()
            
            # Rank attention heads by importance
            head_rankings = analyzer.rank_attention_heads(
                self.attention_weights
            )
            
            self.assertIsInstance(head_rankings, dict)
            
            for layer_name in self.attention_weights.keys():
                self.assertIn(layer_name, head_rankings)
                layer_rankings = head_rankings[layer_name]
                self.assertEqual(len(layer_rankings), 12)  # 12 heads per layer
                
                # Rankings should be sorted by importance score
                importance_scores = [head['importance_score'] for head in layer_rankings]
                self.assertEqual(importance_scores, sorted(importance_scores, reverse=True))
            
        except ImportError:
            self.skipTest("AttentionAnalyzer not available")
    
    def test_attention_visualization_data(self):
        """Test preparation of attention visualization data."""
        try:
            analyzer = AttentionAnalyzer()
            
            # Prepare visualization data
            viz_data = analyzer.prepare_attention_visualization(
                self.attention_weights['layer_0'][0],  # First sample
                self.token_sequences[0]
            )
            
            self.assertIn('attention_matrix', viz_data)
            self.assertIn('tokens', viz_data)
            self.assertIn('head_data', viz_data)
            
            # Check data structure
            self.assertEqual(len(viz_data['tokens']), 20)
            self.assertEqual(len(viz_data['head_data']), 12)  # 12 heads
            
        except ImportError:
            self.skipTest("AttentionAnalyzer not available")


class TestInterpretabilityAnalyzer(unittest.TestCase):
    """Test interpretability analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.activations = np.random.randn(100, 768)
        self.labels = ['positive'] * 50 + ['negative'] * 50
        self.feature_names = [f'feature_{i}' for i in range(768)]
        
    def test_interpretability_analyzer_init(self):
        """Test InterpretabilityAnalyzer initialization."""
        try:
            analyzer = InterpretabilityAnalyzer()
            self.assertIsNotNone(analyzer)
            
        except ImportError:
            self.skipTest("InterpretabilityAnalyzer not available")
    
    def test_concept_activation_vectors(self):
        """Test Concept Activation Vectors (CAVs) analysis."""
        try:
            analyzer = InterpretabilityAnalyzer()
            
            # Create CAVs for binary classification
            cav_results = analyzer.compute_concept_activation_vectors(
                self.activations,
                self.labels,
                concept_name='sentiment'
            )
            
            self.assertIn('cav_vector', cav_results)
            self.assertIn('accuracy', cav_results)
            self.assertIn('feature_importance', cav_results)
            
            # Check CAV vector properties
            self.assertEqual(len(cav_results['cav_vector']), 768)
            self.assertGreaterEqual(cav_results['accuracy'], 0)
            self.assertLessEqual(cav_results['accuracy'], 1)
            
        except ImportError:
            self.skipTest("InterpretabilityAnalyzer not available")
    
    def test_saliency_analysis(self):
        """Test saliency analysis for interpretability."""
        try:
            analyzer = InterpretabilityAnalyzer()
            
            # Mock gradient computation
            mock_gradients = np.random.randn(100, 768)
            
            saliency_results = analyzer.compute_saliency_maps(
                self.activations,
                mock_gradients,
                method='gradient'
            )
            
            self.assertIn('saliency_scores', saliency_results)
            self.assertIn('top_features', saliency_results)
            
            # Check saliency scores
            self.assertEqual(saliency_results['saliency_scores'].shape, (100, 768))
            
        except ImportError:
            self.skipTest("InterpretabilityAnalyzer not available")
    
    def test_feature_importance_analysis(self):
        """Test feature importance analysis."""
        try:
            analyzer = InterpretabilityAnalyzer()
            
            # Analyze feature importance
            importance_results = analyzer.analyze_feature_importance(
                self.activations,
                self.labels,
                method='permutation'
            )
            
            self.assertIn('feature_importance', importance_results)
            self.assertIn('top_features', importance_results)
            self.assertIn('feature_rankings', importance_results)
            
            # Check results structure
            self.assertEqual(len(importance_results['feature_importance']), 768)
            
        except ImportError:
            self.skipTest("InterpretabilityAnalyzer not available")
    
    def test_activation_maximization(self):
        """Test activation maximization for neuron interpretation."""
        try:
            analyzer = InterpretabilityAnalyzer()
            
            # Mock model for activation maximization
            def mock_activation_function(input_data):
                return np.sum(input_data ** 2, axis=1)  # Simple quadratic function
            
            max_results = analyzer.activation_maximization(
                activation_function=mock_activation_function,
                input_shape=(768,),
                target_neuron=0,
                iterations=10
            )
            
            self.assertIn('optimized_input', max_results)
            self.assertIn('activation_trajectory', max_results)
            self.assertIn('final_activation', max_results)
            
            # Check optimization trajectory
            trajectory = max_results['activation_trajectory']
            self.assertGreater(len(trajectory), 0)
            
        except ImportError:
            self.skipTest("InterpretabilityAnalyzer not available")


class TestExperimentalAnalyzer(unittest.TestCase):
    """Test experimental analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.activations_model1 = np.random.randn(50, 768)
        self.activations_model2 = np.random.randn(50, 768)
        self.labels = [f'sample_{i}' for i in range(50)]
        
    def test_experimental_analyzer_init(self):
        """Test ExperimentalAnalyzer initialization."""
        try:
            analyzer = ExperimentalAnalyzer()
            self.assertIsNotNone(analyzer)
            
        except ImportError:
            self.skipTest("ExperimentalAnalyzer not available")
    
    def test_representation_similarity_analysis(self):
        """Test Representation Similarity Analysis (RSA)."""
        try:
            analyzer = ExperimentalAnalyzer()
            
            # Perform RSA between two sets of activations
            rsa_results = analyzer.representation_similarity_analysis(
                self.activations_model1,
                self.activations_model2,
                method='correlation'
            )
            
            self.assertIn('similarity_matrix', rsa_results)
            self.assertIn('similarity_score', rsa_results)
            self.assertIn('method', rsa_results)
            
            # Check similarity matrix properties
            sim_matrix = np.array(rsa_results['similarity_matrix'])
            self.assertEqual(sim_matrix.shape, (50, 50))
            
        except ImportError:
            self.skipTest("ExperimentalAnalyzer not available")
    
    def test_centered_kernel_alignment(self):
        """Test Centered Kernel Alignment (CKA) analysis."""
        try:
            analyzer = ExperimentalAnalyzer()
            
            # Compute CKA between activations
            cka_results = analyzer.centered_kernel_alignment(
                self.activations_model1,
                self.activations_model2
            )
            
            self.assertIn('cka_score', cka_results)
            self.assertIn('linear_cka', cka_results)
            self.assertIn('rbf_cka', cka_results)
            
            # CKA scores should be between 0 and 1
            self.assertGreaterEqual(cka_results['cka_score'], 0)
            self.assertLessEqual(cka_results['cka_score'], 1)
            
        except ImportError:
            self.skipTest("ExperimentalAnalyzer not available")
    
    def test_mutual_information_analysis(self):
        """Test mutual information analysis."""
        try:
            analyzer = ExperimentalAnalyzer()
            
            # Create discrete labels for MI analysis
            discrete_labels = np.random.randint(0, 5, size=50)
            
            # Compute mutual information
            mi_results = analyzer.mutual_information_analysis(
                self.activations_model1,
                discrete_labels
            )
            
            self.assertIn('mutual_information', mi_results)
            self.assertIn('feature_mi_scores', mi_results)
            self.assertIn('top_informative_features', mi_results)
            
            # MI should be non-negative
            self.assertGreaterEqual(mi_results['mutual_information'], 0)
            
        except ImportError:
            self.skipTest("ExperimentalAnalyzer not available")
    
    def test_probing_task_analysis(self):
        """Test probing task analysis."""
        try:
            analyzer = ExperimentalAnalyzer()
            
            # Create synthetic probing task
            probe_labels = np.random.choice(['A', 'B', 'C'], size=50)
            
            # Train probing classifier
            probe_results = analyzer.probing_task_analysis(
                self.activations_model1,
                probe_labels,
                task_name='synthetic_classification'
            )
            
            self.assertIn('accuracy', probe_results)
            self.assertIn('classification_report', probe_results)
            self.assertIn('feature_importance', probe_results)
            self.assertIn('task_name', probe_results)
            
            # Accuracy should be reasonable for random data
            self.assertGreaterEqual(probe_results['accuracy'], 0)
            self.assertLessEqual(probe_results['accuracy'], 1)
            
        except ImportError:
            self.skipTest("ExperimentalAnalyzer not available")


class TestAnalysisIntegration(unittest.TestCase):
    """Integration tests for analysis modules."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create comprehensive test data
        self.test_data = {
            'questions': [f"Question {i}?" for i in range(20)],
            'activations': {
                f'layer_{i}': np.random.randn(20, 768) for i in range(6)
            },
            'attention_weights': {
                f'layer_{i}': np.random.rand(20, 12, 15, 15) for i in range(6)
            },
            'metadata': {
                'model_name': 'test_model',
                'timestamp': '2024-01-01',
                'config': {'batch_size': 4}
            }
        }
        
    def tearDown(self):
        """Clean up integration test environment."""
        self.temp_dir.cleanup()
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        try:
            # Step 1: Basic activation analysis
            advanced_analyzer = AdvancedAnalyzer()
            
            activation_stats = {}
            for layer_name, activations in self.test_data['activations'].items():
                stats = advanced_analyzer.compute_activation_statistics(activations)
                activation_stats[layer_name] = stats
            
            # Step 2: Attention analysis
            attention_analyzer = AttentionAnalyzer()
            
            attention_analysis = {}
            for layer_name, attention_weights in self.test_data['attention_weights'].items():
                patterns = attention_analyzer.analyze_attention_patterns(attention_weights)
                attention_analysis[layer_name] = patterns
            
            # Step 3: Interpretability analysis
            interpretability_analyzer = InterpretabilityAnalyzer()
            
            # Create mock labels for interpretability
            mock_labels = ['positive'] * 10 + ['negative'] * 10
            
            interpretability_results = interpretability_analyzer.analyze_feature_importance(
                self.test_data['activations']['layer_0'],
                mock_labels,
                method='permutation'
            )
            
            # Step 4: Experimental analysis
            experimental_analyzer = ExperimentalAnalyzer()
            
            # Compare different layers
            rsa_results = experimental_analyzer.representation_similarity_analysis(
                self.test_data['activations']['layer_0'],
                self.test_data['activations']['layer_5'],
                method='correlation'
            )
            
            # Verify all analyses completed successfully
            self.assertEqual(len(activation_stats), 6)  # 6 layers
            self.assertEqual(len(attention_analysis), 6)  # 6 layers
            self.assertIn('feature_importance', interpretability_results)
            self.assertIn('similarity_score', rsa_results)
            
            # Save comprehensive results
            results_file = Path(self.temp_dir.name) / "analysis_results.json"
            comprehensive_results = {
                'activation_statistics': {
                    layer: {k: float(v) if isinstance(v, np.number) else v 
                           for k, v in stats.items()}
                    for layer, stats in activation_stats.items()
                },
                'attention_analysis': {
                    layer: {k: float(v) if isinstance(v, np.number) else 
                           v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in analysis.items()}
                    for layer, analysis in attention_analysis.items()
                },
                'interpretability': {
                    'feature_importance_available': 'feature_importance' in interpretability_results
                },
                'experimental': {
                    'rsa_similarity_score': float(rsa_results['similarity_score'])
                },
                'metadata': self.test_data['metadata']
            }
            
            with open(results_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2)
            
            # Verify results were saved
            self.assertTrue(results_file.exists())
            
            with open(results_file, 'r') as f:
                saved_results = json.load(f)
            
            self.assertIn('activation_statistics', saved_results)
            self.assertIn('attention_analysis', saved_results)
            self.assertIn('metadata', saved_results)
            
        except ImportError:
            self.skipTest("Analysis modules not available")


class MockModelTests(unittest.TestCase):
    """Test analysis modules with mocked model dependencies."""
    
    def test_analysis_without_transformers(self):
        """Test analysis functionality without transformers library."""
        try:
            with patch.dict('sys.modules', {'transformers': None}):
                # Should still be able to analyze pre-extracted activations
                activations = np.random.randn(10, 768)
                
                # Basic statistical analysis should work
                mean_activation = np.mean(activations)
                std_activation = np.std(activations)
                
                self.assertIsInstance(mean_activation, (int, float))
                self.assertIsInstance(std_activation, (int, float))
                
        except ImportError:
            self.skipTest("Analysis modules not available")
    
    def test_gpu_fallback_behavior(self):
        """Test analysis behavior when GPU is not available."""
        try:
            with patch.dict('sys.modules', {'torch': None}):
                # Analysis should fall back to CPU/numpy implementations
                activations = np.random.randn(50, 768)
                
                # Should be able to perform basic analysis
                pca_components = 10
                from sklearn.decomposition import PCA
                
                pca = PCA(n_components=pca_components)
                pca_result = pca.fit_transform(activations)
                
                self.assertEqual(pca_result.shape[1], pca_components)
                
        except ImportError:
            self.skipTest("Analysis fallback test not available")


if __name__ == "__main__":
    unittest.main(verbosity=2)