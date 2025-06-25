"""
Comprehensive unit tests for Domain Specialists (Section 4.1).

This module tests the DomainSpecializationFramework and all domain specialists
with focus on question generation quality, terminology density, and classification accuracy.
"""

import pytest
import numpy as np
import hypothesis
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.data_generation.domain_specialists import (
    DomainSpecializationFramework,
    BaseDomainSpecialist,
    ScienceSpecialist,
    LiteratureSpecialist,
    HistorySpecialist,
    MathematicsSpecialist,
    TechnologySpecialist,
    create_domain_specialist,
    get_available_domains,
    generate_domain_specific_questions
)
# Local test utilities
def assert_questions_quality(questions: List[str], min_length: int = 10):
    """Assert that generated questions meet quality standards."""
    assert len(questions) > 0, "Should generate at least one question"
    
    for i, question in enumerate(questions):
        assert isinstance(question, str), f"Question {i} should be a string"
        assert len(question.strip()) >= min_length, f"Question {i} too short: '{question}'"
        assert question.strip().endswith('?'), f"Question {i} should end with '?': '{question}'"
        assert question.strip()[0].isupper(), f"Question {i} should start with capital letter: '{question}'"


class PerformanceTimer:
    """Context manager for measuring test performance."""
    
    def __init__(self, max_duration: float = 1.0):
        self.max_duration = max_duration
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.duration = time.time() - self.start_time
        assert self.duration < self.max_duration, f"Test took {self.duration:.2f}s, expected < {self.max_duration}s"


class TestBaseDomainSpecialist:
    """Test the base domain specialist class."""
    
    def test_base_specialist_abstract_methods(self):
        """Test that BaseDomainSpecialist cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDomainSpecialist()
    
    def test_base_specialist_interface(self):
        """Test that all required methods are defined in the interface."""
        required_methods = [
            'get_domain_vocabulary',
            'get_question_patterns',
            'validate_domain_relevance',
            'generate_questions'
        ]
        
        for method in required_methods:
            assert hasattr(BaseDomainSpecialist, method)
            assert callable(getattr(BaseDomainSpecialist, method))


class TestScienceSpecialist:
    """Test the Science domain specialist."""
    
    @pytest.fixture
    def science_specialist(self):
        """Create a Science specialist for testing."""
        return ScienceSpecialist()
    
    def test_initialization(self, science_specialist):
        """Test Science specialist initialization."""
        assert science_specialist.domain == "science"
        assert len(science_specialist.get_domain_vocabulary()) >= 50
        assert len(science_specialist.get_question_patterns()) >= 10
    
    def test_domain_vocabulary_quality(self, science_specialist):
        """Test that Science vocabulary contains relevant terms."""
        vocab = science_specialist.get_domain_vocabulary()
        
        # Check for expected scientific terms
        expected_terms = ["atom", "molecule", "electron", "photosynthesis", "DNA"]
        for term in expected_terms:
            assert any(term.lower() in v.lower() for v in vocab), f"Missing scientific term: {term}"
        
        # Check vocabulary quality
        assert all(len(term.strip()) > 2 for term in vocab), "All vocabulary terms should be > 2 characters"
        assert all(term.isalpha() or '_' in term or ' ' in term for term in vocab), "Vocabulary should contain valid terms"
    
    def test_question_generation(self, science_specialist):
        """Test Science question generation."""
        with PerformanceTimer(max_duration=1.0):
            questions = science_specialist.generate_questions(count=5)
        
        assert_questions_quality(questions, min_length=15)
        assert len(questions) == 5
        
        # Check for science-specific content
        science_indicators = ["energy", "matter", "experiment", "theory", "research", "study", "analysis"]
        for question in questions:
            assert any(indicator in question.lower() for indicator in science_indicators), \
                f"Question lacks scientific terminology: {question}"
    
    def test_domain_relevance_validation(self, science_specialist):
        """Test domain relevance validation."""
        # Test positive cases
        science_text = "The photosynthesis process converts carbon dioxide into glucose using sunlight energy."
        assert science_specialist.validate_domain_relevance(science_text) >= 0.7
        
        # Test negative cases
        literature_text = "The protagonist's journey reveals deep metaphorical symbolism in the narrative."
        assert science_specialist.validate_domain_relevance(literature_text) < 0.5
        
        # Test edge cases
        assert science_specialist.validate_domain_relevance("") == 0.0
        assert science_specialist.validate_domain_relevance("a") < 0.1
    
    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=10, deadline=2000, suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
    def test_question_generation_property_based(self, science_specialist, count):
        """Property-based test for question generation."""
        questions = science_specialist.generate_questions(count=count)
        
        assert len(questions) == count
        assert_questions_quality(questions)
        
        # All questions should be unique
        assert len(set(questions)) == len(questions), "Generated questions should be unique"


class TestLiteratureSpecialist:
    """Test the Literature domain specialist."""
    
    @pytest.fixture
    def literature_specialist(self):
        """Create a Literature specialist for testing."""
        return LiteratureSpecialist()
    
    def test_initialization(self, literature_specialist):
        """Test Literature specialist initialization."""
        assert literature_specialist.domain == "literature"
        assert len(literature_specialist.get_domain_vocabulary()) >= 50
        assert len(literature_specialist.get_question_patterns()) >= 10
    
    def test_domain_vocabulary_quality(self, literature_specialist):
        """Test that Literature vocabulary contains relevant terms."""
        vocab = literature_specialist.get_domain_vocabulary()
        
        # Check for expected literary terms
        expected_terms = ["metaphor", "symbolism", "narrative", "protagonist", "theme"]
        for term in expected_terms:
            assert any(term.lower() in v.lower() for v in vocab), f"Missing literary term: {term}"
    
    def test_question_generation(self, literature_specialist):
        """Test Literature question generation."""
        questions = literature_specialist.generate_questions(count=3)
        
        assert_questions_quality(questions, min_length=20)
        
        # Check for literature-specific content
        lit_indicators = ["character", "theme", "author", "narrative", "symbolism", "metaphor", "analysis"]
        for question in questions:
            assert any(indicator in question.lower() for indicator in lit_indicators), \
                f"Question lacks literary terminology: {question}"


class TestMathematicsSpecialist:
    """Test the Mathematics domain specialist."""
    
    @pytest.fixture
    def math_specialist(self):
        """Create a Mathematics specialist for testing."""
        return MathematicsSpecialist()
    
    def test_mathematical_terminology(self, math_specialist):
        """Test that Mathematics specialist uses proper mathematical terms."""
        vocab = math_specialist.get_domain_vocabulary()
        
        # Check for mathematical terms
        math_terms = ["equation", "derivative", "integral", "theorem", "proof", "function"]
        for term in math_terms:
            assert any(term.lower() in v.lower() for v in vocab), f"Missing mathematical term: {term}"
    
    def test_question_complexity(self, math_specialist):
        """Test that mathematical questions have appropriate complexity."""
        questions = math_specialist.generate_questions(count=5)
        
        # Mathematical questions should often contain numbers or mathematical notation
        math_patterns = ["=", "+", "-", "×", "÷", "∫", "∂", "Σ", r"\d+"]
        
        math_question_count = 0
        for question in questions:
            if any(pattern in question for pattern in math_patterns[:-1]) or \
               any(char.isdigit() for char in question):
                math_question_count += 1
        
        # At least 60% of questions should contain mathematical elements
        assert math_question_count >= len(questions) * 0.6, \
            f"Only {math_question_count}/{len(questions)} questions contain mathematical elements"


class TestDomainSpecializationFramework:
    """Test the overall Domain Specialization Framework."""
    
    @pytest.fixture
    def framework(self):
        """Create a DomainSpecializationFramework for testing."""
        return DomainSpecializationFramework()
    
    def test_framework_initialization(self, framework):
        """Test framework initialization."""
        assert hasattr(framework, 'specialists')
        assert len(framework.specialists) >= 20  # Should have all 20 specialists
    
    def test_get_available_domains_function(self):
        """Test the get_available_domains utility function."""
        domains = get_available_domains()
        
        assert len(domains) >= 20
        assert "science" in domains
        assert "literature" in domains
        assert "mathematics" in domains
        assert "technology" in domains
        assert "history" in domains
    
    def test_create_domain_specialist_function(self):
        """Test the create_domain_specialist factory function."""
        # Test valid domains
        science_specialist = create_domain_specialist("science")
        assert science_specialist.domain == "science"
        assert isinstance(science_specialist, ScienceSpecialist)
        
        literature_specialist = create_domain_specialist("literature")
        assert literature_specialist.domain == "literature"
        assert isinstance(literature_specialist, LiteratureSpecialist)
        
        # Test invalid domain
        with pytest.raises(ValueError, match="Unknown domain"):
            create_domain_specialist("invalid_domain")
    
    def test_all_specialists_functional(self, framework):
        """Test that all 20 specialists are functional and generate questions."""
        domains = get_available_domains()
        
        for domain in domains:
            specialist = create_domain_specialist(domain)
            
            # Test that specialist can generate questions
            with PerformanceTimer(max_duration=2.0):
                questions = specialist.generate_questions(count=2)
            
            assert len(questions) == 2, f"Domain {domain} failed to generate questions"
            assert_questions_quality(questions, min_length=10)
    
    def test_cross_domain_contamination_prevention(self):
        """Test that specialists don't generate questions from other domains."""
        # Create specialists for testing
        science_specialist = create_domain_specialist("science")
        literature_specialist = create_domain_specialist("literature")
        
        # Generate questions
        science_questions = science_specialist.generate_questions(count=10)
        literature_questions = literature_specialist.generate_questions(count=10)
        
        # Test domain relevance
        for question in science_questions:
            science_relevance = science_specialist.validate_domain_relevance(question)
            literature_relevance = literature_specialist.validate_domain_relevance(question)
            
            # Science questions should be more relevant to science than literature
            assert science_relevance > literature_relevance, \
                f"Science question '{question}' has higher literature relevance"
    
    def test_generate_domain_specific_questions_function(self):
        """Test the generate_domain_specific_questions utility function."""
        questions_dict = generate_domain_specific_questions(
            domains=["science", "literature", "mathematics"],
            questions_per_domain=3
        )
        
        assert len(questions_dict) == 3
        assert "science" in questions_dict
        assert "literature" in questions_dict
        assert "mathematics" in questions_dict
        
        for domain, questions in questions_dict.items():
            assert len(questions) == 3
            assert_questions_quality(questions)
    
    @pytest.mark.slow
    def test_performance_benchmarks(self, framework):
        """Test performance benchmarks for the framework."""
        domains = get_available_domains()
        
        # Test bulk question generation performance
        with PerformanceTimer(max_duration=10.0):
            all_questions = generate_domain_specific_questions(
                domains=domains[:10],  # Test first 10 domains
                questions_per_domain=5
            )
        
        assert len(all_questions) == 10
        total_questions = sum(len(questions) for questions in all_questions.values())
        assert total_questions == 50


class TestTerminologyDensityAndClassification:
    """Test terminology density and classification accuracy."""
    
    def test_terminology_density_calculation(self):
        """Test calculation of terminology density in questions."""
        science_specialist = create_domain_specialist("science")
        
        # Generate questions and calculate terminology density
        questions = science_specialist.generate_questions(count=10)
        vocab = science_specialist.get_domain_vocabulary()
        
        for question in questions:
            # Calculate terminology density - use exact word matches
            words = [word.strip('.,!?:;') for word in question.lower().split()]
            vocab_lower = [term.lower() for term in vocab]
            terminology_count = sum(1 for word in words if word in vocab_lower)
            density = terminology_count / len(words) if words else 0
            
            # Terminology density should be reasonable (5-60%)
            assert 0.05 <= density <= 0.6, \
                f"Terminology density {density:.2f} outside expected range for: {question}"
    
    def test_classification_accuracy_basic(self):
        """Test basic classification accuracy between domains."""
        # Create specialists
        science_specialist = create_domain_specialist("science")
        literature_specialist = create_domain_specialist("literature")
        
        # Test clear domain examples
        science_text = "Photosynthesis converts carbon dioxide and water into glucose using chlorophyll."
        literature_text = "The protagonist's internal monologue reveals the narrative's central metaphor."
        
        # Science text should score higher with science specialist
        science_score_for_science = science_specialist.validate_domain_relevance(science_text)
        literature_score_for_science = literature_specialist.validate_domain_relevance(science_text)
        assert science_score_for_science > literature_score_for_science
        
        # Literature text should score higher with literature specialist
        science_score_for_literature = science_specialist.validate_domain_relevance(literature_text)
        literature_score_for_literature = literature_specialist.validate_domain_relevance(literature_text)
        assert literature_score_for_literature > science_score_for_literature


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in domain specialists."""
    
    def test_empty_input_handling(self):
        """Test handling of empty or invalid inputs."""
        science_specialist = create_domain_specialist("science")
        
        # Test empty string
        assert science_specialist.validate_domain_relevance("") == 0.0
        
        # Test whitespace-only string
        assert science_specialist.validate_domain_relevance("   ") == 0.0
        
        # Test very short string
        relevance = science_specialist.validate_domain_relevance("a")
        assert 0.0 <= relevance <= 0.1
    
    def test_zero_question_generation(self):
        """Test generation of zero questions."""
        science_specialist = create_domain_specialist("science")
        questions = science_specialist.generate_questions(count=0)
        assert questions == []
    
    def test_large_question_generation(self):
        """Test generation of large numbers of questions."""
        science_specialist = create_domain_specialist("science")
        
        with PerformanceTimer(max_duration=5.0):
            questions = science_specialist.generate_questions(count=50)
        
        assert len(questions) == 50
        assert_questions_quality(questions)
        
        # Check that we get some variety in questions
        unique_questions = set(questions)
        uniqueness_ratio = len(unique_questions) / len(questions)
        assert uniqueness_ratio >= 0.8, f"Only {uniqueness_ratio:.2f} of questions are unique"
    
    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=20, deadline=1000)
    def test_domain_relevance_robustness(self, text):
        """Property-based test for domain relevance validation robustness."""
        science_specialist = create_domain_specialist("science")
        
        try:
            relevance = science_specialist.validate_domain_relevance(text)
            assert 0.0 <= relevance <= 1.0, f"Relevance score {relevance} outside valid range [0,1]"
            assert isinstance(relevance, (int, float)), f"Relevance score should be numeric, got {type(relevance)}"
        except Exception as e:
            # Should not raise unexpected exceptions
            pytest.fail(f"Domain relevance validation failed on input '{text[:50]}...': {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
