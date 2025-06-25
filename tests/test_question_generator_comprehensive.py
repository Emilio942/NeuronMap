"""
Comprehensive unit tests for Question Generator.

This module tests the QuestionGenerator class and related functionality
with focus on question quality, difficulty assessment, and generation performance.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.data_generation.question_generator import QuestionGenerator
from src.data_generation.difficulty_assessment import DifficultyAssessmentEngine
from tests.conftest import assert_questions_quality, PerformanceTimer


class TestQuestionGenerator:
    """Test the QuestionGenerator class."""
    
    @pytest.fixture
    def question_generator(self):
        """Create a QuestionGenerator for testing."""
        return QuestionGenerator()
    
    @pytest.fixture
    def mock_difficulty_engine(self):
        """Create a mock difficulty assessment engine."""
        engine = Mock(spec=DifficultyAssessmentEngine)
        engine.assess_difficulty.return_value = 0.5
        engine.assess_question_quality.return_value = {
            "clarity": 0.8,
            "specificity": 0.7,
            "complexity": 0.6,
            "overall_score": 0.7
        }
        return engine
    
    def test_initialization(self, question_generator):
        """Test QuestionGenerator initialization."""
        assert hasattr(question_generator, 'difficulty_engine')
        assert hasattr(question_generator, 'domain_specialists')
    
    def test_basic_question_generation(self, question_generator):
        """Test basic question generation functionality."""
        with PerformanceTimer(max_duration=2.0):
            questions = question_generator.generate_questions(count=5)
        
        assert len(questions) == 5
        assert_questions_quality(questions, min_length=10)
        
        # All questions should be strings
        assert all(isinstance(q, str) for q in questions)
        
        # Questions should be non-empty after stripping
        assert all(len(q.strip()) > 0 for q in questions)
    
    def test_domain_specific_generation(self, question_generator):
        """Test domain-specific question generation."""
        # Test science domain
        science_questions = question_generator.generate_questions(
            count=3, 
            domain="science"
        )
        
        assert len(science_questions) == 3
        assert_questions_quality(science_questions)
        
        # Should contain science-related terms
        science_terms = ["energy", "matter", "experiment", "theory", "analysis", "research"]
        for question in science_questions:
            assert any(term in question.lower() for term in science_terms), \
                f"Science question lacks scientific terminology: {question}"
    
    def test_difficulty_based_generation(self, question_generator):
        """Test generation of questions with specific difficulty levels."""
        # Generate easy questions (difficulty 1-3)
        easy_questions = question_generator.generate_questions(
            count=3,
            difficulty_range=(1, 3)
        )
        
        # Generate hard questions (difficulty 7-10)
        hard_questions = question_generator.generate_questions(
            count=3,
            difficulty_range=(7, 10)
        )
        
        assert len(easy_questions) == 3
        assert len(hard_questions) == 3
        
        assert_questions_quality(easy_questions)
        assert_questions_quality(hard_questions)
        
        # Hard questions should generally be longer and more complex
        avg_easy_length = np.mean([len(q.split()) for q in easy_questions])
        avg_hard_length = np.mean([len(q.split()) for q in hard_questions])
        
        # This is a heuristic - harder questions tend to be longer
        assert avg_hard_length >= avg_easy_length * 0.8, \
            f"Hard questions ({avg_hard_length:.1f} words) should be similar length or longer than easy questions ({avg_easy_length:.1f} words)"
    
    def test_question_uniqueness(self, question_generator):
        """Test that generated questions are unique."""
        questions = question_generator.generate_questions(count=20)
        
        unique_questions = set(questions)
        uniqueness_ratio = len(unique_questions) / len(questions)
        
        # At least 90% of questions should be unique
        assert uniqueness_ratio >= 0.9, \
            f"Only {uniqueness_ratio:.2%} of questions are unique"
    
    @patch('src.data_generation.question_generator.QuestionGenerator._generate_with_ollama')
    def test_ollama_integration(self, mock_ollama, question_generator):
        """Test integration with Ollama API."""
        # Mock Ollama responses
        mock_ollama.return_value = [
            "What is the capital of France?",
            "How do neural networks learn?",
            "Explain photosynthesis in plants."
        ]
        
        questions = question_generator.generate_questions(
            count=3,
            use_ollama=True
        )
        
        assert len(questions) == 3
        assert_questions_quality(questions)
        mock_ollama.assert_called_once()
    
    def test_quality_assessment_integration(self, question_generator, mock_difficulty_engine):
        """Test integration with difficulty assessment engine."""
        # Patch the difficulty engine
        question_generator.difficulty_engine = mock_difficulty_engine
        
        questions = question_generator.generate_questions(count=3)
        quality_scores = [
            question_generator.assess_question_quality(q) for q in questions
        ]
        
        assert len(quality_scores) == 3
        
        # Each quality score should have expected structure
        for score in quality_scores:
            assert isinstance(score, dict)
            assert "overall_score" in score
            assert 0.0 <= score["overall_score"] <= 1.0
    
    def test_batch_generation_performance(self, question_generator):
        """Test performance of batch question generation."""
        with PerformanceTimer(max_duration=5.0):
            questions = question_generator.generate_questions(count=50)
        
        assert len(questions) == 50
        assert_questions_quality(questions)
    
    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=10, deadline=3000)
    def test_question_count_property(self, question_generator, count):
        """Property-based test for question count handling."""
        questions = question_generator.generate_questions(count=count)
        
        assert len(questions) == count
        assert_questions_quality(questions)
    
    def test_error_handling_invalid_domain(self, question_generator):
        """Test error handling for invalid domain."""
        with pytest.raises(ValueError, match="Unknown domain"):
            question_generator.generate_questions(
                count=3,
                domain="invalid_domain_name"
            )
    
    def test_error_handling_invalid_difficulty(self, question_generator):
        """Test error handling for invalid difficulty range."""
        # Difficulty range should be between 1 and 10
        with pytest.raises(ValueError, match="Difficulty.*range"):
            question_generator.generate_questions(
                count=3,
                difficulty_range=(15, 20)  # Invalid range
            )
        
        with pytest.raises(ValueError, match="Difficulty.*range"):
            question_generator.generate_questions(
                count=3,
                difficulty_range=(5, 3)  # Min > max
            )
    
    def test_zero_questions_generation(self, question_generator):
        """Test generation of zero questions."""
        questions = question_generator.generate_questions(count=0)
        assert questions == []


class TestDifficultyAssessmentIntegration:
    """Test integration with difficulty assessment engine."""
    
    @pytest.fixture
    def difficulty_engine(self):
        """Create a DifficultyAssessmentEngine for testing."""
        return DifficultyAssessmentEngine()
    
    def test_difficulty_engine_initialization(self, difficulty_engine):
        """Test DifficultyAssessmentEngine initialization."""
        assert hasattr(difficulty_engine, 'assess_difficulty')
        assert hasattr(difficulty_engine, 'assess_question_quality')
    
    def test_difficulty_assessment_ranges(self, difficulty_engine):
        """Test that difficulty assessment returns valid ranges."""
        test_questions = [
            "What is 2 + 2?",  # Easy
            "Explain the significance of quantum entanglement in modern physics.",  # Hard
            "How do you make a sandwich?",  # Medium
        ]
        
        for question in test_questions:
            difficulty = difficulty_engine.assess_difficulty(question)
            
            assert isinstance(difficulty, (int, float))
            assert 0.0 <= difficulty <= 1.0, \
                f"Difficulty {difficulty} outside valid range [0,1] for: {question}"
    
    def test_quality_assessment_structure(self, difficulty_engine):
        """Test quality assessment returns proper structure."""
        question = "What are the main principles of machine learning?"
        quality = difficulty_engine.assess_question_quality(question)
        
        assert isinstance(quality, dict)
        
        # Expected quality metrics
        expected_keys = ["clarity", "specificity", "complexity", "overall_score"]
        for key in expected_keys:
            assert key in quality, f"Missing quality metric: {key}"
            assert 0.0 <= quality[key] <= 1.0, \
                f"Quality metric {key} = {quality[key]} outside valid range [0,1]"
    
    def test_difficulty_consistency(self, difficulty_engine):
        """Test that difficulty assessment is consistent."""
        question = "Explain the process of cellular respiration."
        
        # Multiple assessments of the same question should be similar
        difficulties = [
            difficulty_engine.assess_difficulty(question) 
            for _ in range(5)
        ]
        
        # Standard deviation should be small (consistent)
        std_dev = np.std(difficulties)
        assert std_dev < 0.1, f"Difficulty assessment inconsistent: std={std_dev:.3f}"
    
    def test_relative_difficulty_ordering(self, difficulty_engine):
        """Test that relative difficulty ordering makes sense."""
        easy_question = "What color is the sky?"
        medium_question = "How do computers process information?"
        hard_question = "Derive the Schrödinger equation from first principles."
        
        easy_diff = difficulty_engine.assess_difficulty(easy_question)
        medium_diff = difficulty_engine.assess_difficulty(medium_question)
        hard_diff = difficulty_engine.assess_difficulty(hard_question)
        
        # Should maintain relative ordering
        assert easy_diff <= medium_diff, \
            f"Easy question ({easy_diff:.2f}) should be easier than medium ({medium_diff:.2f})"
        assert medium_diff <= hard_diff, \
            f"Medium question ({medium_diff:.2f}) should be easier than hard ({hard_diff:.2f})"


class TestQuestionGeneratorEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def question_generator(self):
        """Create a QuestionGenerator for testing."""
        return QuestionGenerator()
    
    def test_empty_input_handling(self, question_generator):
        """Test handling of empty or minimal inputs."""
        # Zero questions
        questions = question_generator.generate_questions(count=0)
        assert questions == []
        
        # Empty domain (should use default)
        questions = question_generator.generate_questions(count=1, domain=None)
        assert len(questions) == 1
    
    def test_large_batch_generation(self, question_generator):
        """Test generation of large batches of questions."""
        with PerformanceTimer(max_duration=10.0):
            questions = question_generator.generate_questions(count=100)
        
        assert len(questions) == 100
        assert_questions_quality(questions)
        
        # Check diversity in large batches
        unique_questions = set(questions)
        diversity_ratio = len(unique_questions) / len(questions)
        assert diversity_ratio >= 0.7, \
            f"Large batch has low diversity: {diversity_ratio:.2%}"
    
    def test_memory_usage_large_batches(self, question_generator):
        """Test memory usage doesn't grow excessively with large batches."""
        import gc
        
        # Generate questions in batches and check memory doesn't grow
        for batch_size in [10, 50, 100]:
            gc.collect()
            initial_objects = len(gc.get_objects())
            
            questions = question_generator.generate_questions(count=batch_size)
            del questions
            
            gc.collect()
            final_objects = len(gc.get_objects())
            
            # Memory growth should be reasonable
            growth = final_objects - initial_objects
            assert growth < batch_size * 10, \
                f"Excessive memory growth: {growth} objects for batch size {batch_size}"
    
    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=10, deadline=2000)
    def test_domain_name_robustness(self, question_generator, domain_name):
        """Property-based test for domain name handling."""
        try:
            # Most domain names should either work or raise ValueError
            questions = question_generator.generate_questions(count=1, domain=domain_name)
            assert isinstance(questions, list)
        except ValueError:
            # Expected for invalid domain names
            pass
        except Exception as e:
            pytest.fail(f"Unexpected exception for domain '{domain_name}': {e}")
    
    def test_concurrent_generation(self, question_generator):
        """Test thread safety of question generation."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def generate_questions():
            try:
                questions = question_generator.generate_questions(count=5)
                results.put(("success", questions))
            except Exception as e:
                results.put(("error", str(e)))
        
        # Start multiple threads
        threads = [threading.Thread(target=generate_questions) for _ in range(3)]
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check all threads completed successfully
        success_count = 0
        while not results.empty():
            status, result = results.get()
            if status == "success":
                success_count += 1
                assert len(result) == 5
                assert_questions_quality(result)
            else:
                pytest.fail(f"Thread failed with error: {result}")
        
        assert success_count == 3, "All threads should complete successfully"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
