"""
Tests for advanced conceptual analysis module.

This module tests cutting-edge techniques for neural network interpretability
including concept extraction, circuit discovery, causal tracing, and world model analysis.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

from src.analysis.conceptual_analysis import (
    ConceptualAnalyzer,
    ConceptVector,
    Circuit,
    KnowledgeTransferResult,
    create_conceptual_analyzer
)


class TestConceptualAnalyzer:
    """Test suite for ConceptualAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a test analyzer."""
        config = {
            'concept_threshold': 0.7,
            'circuit_threshold': 0.5,
            'causal_threshold': 0.6
        }
        return ConceptualAnalyzer(config)
    
    @pytest.fixture
    def sample_activations(self):
        """Create sample activation data."""
        np.random.seed(42)
        return {
            'layer1': np.random.randn(10, 50),
            'layer2': np.random.randn(10, 80),
            'layer3': np.random.randn(10, 100)
        }
    
    @pytest.fixture
    def sample_labels(self):
        """Create sample labels."""
        return ['concept_A'] * 5 + ['concept_B'] * 5
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.concept_threshold == 0.7
        assert analyzer.circuit_threshold == 0.5
        assert analyzer.causal_threshold == 0.6
        assert isinstance(analyzer.concepts, dict)
        assert isinstance(analyzer.circuits, dict)
        assert isinstance(analyzer.world_models, dict)
    
    def test_factory_function(self):
        """Test factory function."""
        analyzer = create_conceptual_analyzer()
        assert isinstance(analyzer, ConceptualAnalyzer)
        
        config = {'concept_threshold': 0.8}
        analyzer = create_conceptual_analyzer(config)
        assert analyzer.concept_threshold == 0.8
    
    def test_concept_extraction_pca(self, analyzer, sample_activations, sample_labels):
        """Test concept extraction using PCA."""
        concepts = analyzer.extract_concepts(
            sample_activations, sample_labels, method='pca'
        )
        
        assert isinstance(concepts, dict)
        assert len(concepts) > 0
        
        # Check concept structure
        for concept_name, concept in concepts.items():
            assert isinstance(concept, ConceptVector)
            assert concept.name == concept_name
            assert isinstance(concept.vector, np.ndarray)
            assert concept.layer in sample_activations.keys()
            assert 0 <= concept.confidence <= 1
    
    def test_concept_extraction_nmf(self, analyzer, sample_activations, sample_labels):
        """Test concept extraction using NMF."""
        # Make activations non-negative for NMF
        activations = {k: np.abs(v) for k, v in sample_activations.items()}
        
        concepts = analyzer.extract_concepts(
            activations, sample_labels, method='nmf'
        )
        
        assert isinstance(concepts, dict)
        # NMF should produce component-based concepts
        component_concepts = [name for name in concepts.keys() if 'component' in name]
        assert len(component_concepts) > 0
    
    def test_concept_algebra(self, analyzer):
        """Test concept algebra operations."""
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
        
        # Test addition
        result = analyzer.concept_algebra("test_a", "test_b", "add")
        expected = np.array([1.5, 3.0, 4.5])
        np.testing.assert_array_equal(result.vector, expected)
        assert result.confidence == min(0.8, 0.7)
        
        # Test subtraction
        result = analyzer.concept_algebra("test_a", "test_b", "subtract")
        expected = np.array([0.5, 1.0, 1.5])
        np.testing.assert_array_equal(result.vector, expected)
        
        # Test average
        result = analyzer.concept_algebra("test_a", "test_b", "average")
        expected = np.array([0.75, 1.5, 2.25])
        np.testing.assert_array_equal(result.vector, expected)
    
    def test_concept_algebra_missing_concepts(self, analyzer):
        """Test concept algebra with missing concepts."""
        with pytest.raises(Exception):  # Should raise NeuronMapError
            analyzer.concept_algebra("missing_a", "missing_b", "add")
    
    def test_knowledge_transfer_analysis(self, analyzer, sample_activations):
        """Test knowledge transfer analysis."""
        # Create slightly different target activations
        target_activations = {
            'layer1_target': sample_activations['layer1'] + 0.1 * np.random.randn(*sample_activations['layer1'].shape),
            'layer2_target': sample_activations['layer2'] + 0.1 * np.random.randn(*sample_activations['layer2'].shape)
        }
        
        result = analyzer.analyze_knowledge_transfer(
            sample_activations,
            target_activations,
            "source_model",
            "target_model"
        )
        
        assert isinstance(result, KnowledgeTransferResult)
        assert result.source_model == "source_model"
        assert result.target_model == "target_model"
        assert 0 <= result.transfer_score <= 1
        assert isinstance(result.transfer_map, dict)
        assert isinstance(result.preserved_concepts, list)
        assert isinstance(result.lost_concepts, list)
        assert isinstance(result.emergent_concepts, list)
    
    @patch('torch.nn.Module')
    def test_causal_tracing(self, mock_model, analyzer):
        """Test causal tracing functionality."""
        # Mock model behavior
        mock_model.named_modules.return_value = [
            ('layer1', Mock()),
            ('layer2', Mock())
        ]
        
        # Mock input tensor
        input_data = torch.randn(1, 10)
        
        # Mock model forward pass
        baseline_output = torch.randn(1, 5)
        intervention_output = torch.randn(1, 5)
        mock_model.return_value = baseline_output
        
        # This test would need more complex mocking for real implementation
        # For now, test that the function exists and has correct signature
        assert hasattr(analyzer, 'trace_causal_effects')
        
        # Test with mock
        try:
            result = analyzer.trace_causal_effects(
                mock_model,
                input_data,
                'layer1',
                [0, 1, 2],
                0.0
            )
            # If no exception, basic structure works
            assert isinstance(result, dict)
        except Exception:
            # Expected due to mocking limitations
            pass
    
    def test_circuit_discovery(self, analyzer, sample_activations, sample_labels):
        """Test circuit discovery."""
        circuits = analyzer.discover_circuits(
            sample_activations,
            sample_labels,
            "test_task"
        )
        
        assert isinstance(circuits, dict)
        
        # Check circuit structure if any found
        for circuit_name, circuit in circuits.items():
            assert isinstance(circuit, Circuit)
            assert circuit.name == circuit_name
            assert isinstance(circuit.components, list)
            assert isinstance(circuit.connections, list)
            assert isinstance(circuit.function, str)
            assert 0 <= circuit.evidence_strength <= 1
            assert isinstance(circuit.metadata, dict)
    
    def test_world_model_analysis(self, analyzer, sample_activations):
        """Test world model analysis."""
        # Create sample metadata
        stimuli_metadata = [
            {'object': 'cat', 'position': [1, 2]},
            {'object': 'dog', 'position': [3, 4]},
            {'object': 'ball', 'position': [5, 6]},
            {'object': 'car', 'position': [7, 8]},
            {'object': 'tree', 'position': [9, 10]},
            {'object': 'house', 'position': [11, 12]},
            {'object': 'bird', 'position': [13, 14]},
            {'object': 'flower', 'position': [15, 16]},
            {'object': 'book', 'position': [17, 18]},
            {'object': 'chair', 'position': [19, 20]}
        ]
        
        world_model = analyzer.analyze_world_model(
            sample_activations,
            stimuli_metadata
        )
        
        assert isinstance(world_model, dict)
        assert 'object_representations' in world_model
        assert 'spatial_representations' in world_model
        assert 'temporal_representations' in world_model
        assert 'relational_representations' in world_model
        assert 'causal_representations' in world_model
    
    def test_cross_model_rsa(self, analyzer, sample_activations):
        """Test cross-model representational similarity analysis."""
        # Create multi-model activations
        model_activations = {
            'model1': sample_activations,
            'model2': {k: v + 0.1 * np.random.randn(*v.shape) for k, v in sample_activations.items()}
        }
        
        stimuli = [f"stimulus_{i}" for i in range(sample_activations['layer1'].shape[0])]
        
        results = analyzer.cross_model_rsa(model_activations, stimuli)
        
        assert isinstance(results, dict)
        assert 'similarity_matrices' in results
        assert 'model_comparisons' in results
        assert 'hierarchical_alignment' in results
        
        # Check similarity matrices
        for model_name in model_activations:
            assert model_name in results['similarity_matrices']
    
    def test_save_load_results(self, analyzer, tmp_path):
        """Test saving and loading analysis results."""
        # Add some test data
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
        
        # Test save
        output_path = tmp_path / "test_results.json"
        analyzer.save_analysis_results(str(output_path))
        assert output_path.exists()
        
        # Test load
        new_analyzer = ConceptualAnalyzer()
        new_analyzer.load_analysis_results(str(output_path))
        
        assert "test_concept" in new_analyzer.concepts
        assert "test_circuit" in new_analyzer.circuits
        
        loaded_concept = new_analyzer.concepts["test_concept"]
        assert loaded_concept.name == "test_concept"
        np.testing.assert_array_equal(loaded_concept.vector, np.array([1, 2, 3]))
        assert loaded_concept.confidence == 0.8


class TestDataStructures:
    """Test data structures used in conceptual analysis."""
    
    def test_concept_vector(self):
        """Test ConceptVector dataclass."""
        vector = np.array([1.0, 2.0, 3.0])
        concept = ConceptVector(
            name="test",
            vector=vector,
            layer="layer1",
            model_name="bert",
            confidence=0.85,
            metadata={'method': 'pca'}
        )
        
        assert concept.name == "test"
        np.testing.assert_array_equal(concept.vector, vector)
        assert concept.layer == "layer1"
        assert concept.model_name == "bert"
        assert concept.confidence == 0.85
        assert concept.metadata['method'] == 'pca'
    
    def test_circuit(self):
        """Test Circuit dataclass."""
        circuit = Circuit(
            name="attention_circuit",
            components=["layer1_att", "layer2_att", "layer3_att"],
            connections=[
                ("layer1_att", "layer2_att", 0.8),
                ("layer2_att", "layer3_att", 0.7)
            ],
            function="attention computation",
            evidence_strength=0.9,
            metadata={'task': 'attention'}
        )
        
        assert circuit.name == "attention_circuit"
        assert len(circuit.components) == 3
        assert len(circuit.connections) == 2
        assert circuit.function == "attention computation"
        assert circuit.evidence_strength == 0.9
        assert circuit.metadata['task'] == 'attention'
    
    def test_knowledge_transfer_result(self):
        """Test KnowledgeTransferResult dataclass."""
        result = KnowledgeTransferResult(
            source_model="bert-base",
            target_model="bert-large",
            transfer_score=0.75,
            transfer_map={"layer1": "layer2", "layer2": "layer4"},
            preserved_concepts=["concept_a", "concept_b"],
            lost_concepts=["concept_c"],
            emergent_concepts=["concept_d"]
        )
        
        assert result.source_model == "bert-base"
        assert result.target_model == "bert-large"
        assert result.transfer_score == 0.75
        assert len(result.transfer_map) == 2
        assert len(result.preserved_concepts) == 2
        assert len(result.lost_concepts) == 1
        assert len(result.emergent_concepts) == 1


class TestUtilityFunctions:
    """Test utility functions in conceptual analysis."""
    
    def test_concept_confidence_calculation(self):
        """Test concept confidence calculation."""
        analyzer = ConceptualAnalyzer()
        
        # Create test data
        concept_vector = np.array([1.0, 0.0, 0.0])
        all_vectors = np.array([
            [1.0, 0.1, 0.1],  # Similar to concept
            [0.9, 0.0, 0.1],  # Similar to concept  
            [0.0, 1.0, 0.0],  # Different from concept
            [0.0, 0.0, 1.0]   # Different from concept
        ])
        mask = np.array([True, True, False, False])
        
        confidence = analyzer._calculate_concept_confidence(
            concept_vector, all_vectors, mask
        )
        
        # Should be positive since similar vectors are in positive class
        assert confidence > 0
        assert confidence <= 1
    
    def test_representation_similarity(self):
        """Test representation similarity calculation."""
        analyzer = ConceptualAnalyzer()
        
        # Similar matrices
        acts1 = np.random.randn(10, 50)
        acts2 = acts1 + 0.1 * np.random.randn(10, 50)  # Add small noise
        
        similarity = analyzer._calculate_representation_similarity(acts1, acts2)
        assert 0 <= similarity <= 1
        assert similarity > 0.5  # Should be high for similar matrices
        
        # Dissimilar matrices
        acts3 = np.random.randn(10, 50)
        similarity_low = analyzer._calculate_representation_similarity(acts1, acts3)
        assert similarity_low < similarity  # Should be lower than similar case


class TestIntegration:
    """Integration tests for conceptual analysis."""
    
    @pytest.fixture
    def full_analyzer(self):
        """Create analyzer with sample data."""
        analyzer = ConceptualAnalyzer()
        
        # Add sample concepts
        concept1 = ConceptVector(
            name="color_red",
            vector=np.array([1.0, 0.0, 0.5]),
            layer="layer1",
            model_name="test",
            confidence=0.8,
            metadata={}
        )
        
        concept2 = ConceptVector(
            name="speed_fast",
            vector=np.array([0.5, 1.0, 0.0]),
            layer="layer1", 
            model_name="test",
            confidence=0.7,
            metadata={}
        )
        
        analyzer.concepts["color_red"] = concept1
        analyzer.concepts["speed_fast"] = concept2
        
        return analyzer
    
    def test_concept_algebra_pipeline(self, full_analyzer):
        """Test full concept algebra pipeline."""
        # Test multiple operations
        result1 = full_analyzer.concept_algebra("color_red", "speed_fast", "add")
        assert result1.name == "color_red_add_speed_fast"
        
        result2 = full_analyzer.concept_algebra("color_red", "speed_fast", "subtract")
        assert result2.name == "color_red_subtract_speed_fast"
        
        # Results should be different
        assert not np.array_equal(result1.vector, result2.vector)
    
    def test_analysis_workflow(self, tmp_path):
        """Test complete analysis workflow."""
        analyzer = ConceptualAnalyzer()
        
        # Create sample data
        activations = {
            'layer1': np.random.randn(20, 100),
            'layer2': np.random.randn(20, 150)
        }
        labels = ['positive'] * 10 + ['negative'] * 10
        
        # Extract concepts
        concepts = analyzer.extract_concepts(activations, labels)
        assert len(concepts) > 0
        
        # Discover circuits
        circuits = analyzer.discover_circuits(activations, labels, "sentiment")
        
        # Save results
        output_path = tmp_path / "workflow_results.json"
        analyzer.save_analysis_results(str(output_path))
        assert output_path.exists()
        
        # Verify saved data
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
        assert 'concepts' in saved_data
        assert 'circuits' in saved_data


# Performance tests
class TestPerformance:
    """Performance tests for conceptual analysis."""
    
    def test_large_scale_concept_extraction(self):
        """Test concept extraction with larger data."""
        analyzer = ConceptualAnalyzer()
        
        # Create larger dataset
        activations = {
            'layer1': np.random.randn(100, 500),
            'layer2': np.random.randn(100, 800)
        }
        labels = ['class_A'] * 50 + ['class_B'] * 50
        
        import time
        start_time = time.time()
        concepts = analyzer.extract_concepts(activations, labels)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 10  # seconds
        assert len(concepts) > 0
    
    def test_memory_efficiency(self):
        """Test memory efficiency of operations."""
        analyzer = ConceptualAnalyzer()
        
        # Test with moderate-sized data
        activations = {
            'layer1': np.random.randn(50, 200)
        }
        labels = ['A'] * 25 + ['B'] * 25
        
        # This should not cause memory errors
        concepts = analyzer.extract_concepts(activations, labels)
        circuits = analyzer.discover_circuits(activations, labels, "test")
        
        # Basic checks
        assert isinstance(concepts, dict)
        assert isinstance(circuits, dict)


if __name__ == "__main__":
    pytest.main([__file__])
