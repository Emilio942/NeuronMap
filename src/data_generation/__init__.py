"""Data generation module for NeuronMap."""

# Import main classes for easy access
from .question_generator import QuestionGenerator

# Try to import optional modules
try:
    from .enhanced_question_generator import EnhancedQuestionGenerator
except ImportError:
    pass

try:
    from .domain_specific_generator import DomainSpecificGenerator
except ImportError:
    pass

# Import difficulty analyzer
try:
    from .difficulty_analyzer import (
        DifficultyAssessmentEngine, DifficultyMetrics, DifficultyLevel,
        ReasoningType, LinguisticFeatures, CognitiveLoadMetrics,
        SemanticComplexity, assess_question_difficulty, batch_assess_difficulty,
        get_difficulty_summary
    )
except ImportError:
    pass

# Export public API
__all__ = [
    'QuestionGenerator',
    'EnhancedQuestionGenerator',
    'DomainSpecificGenerator'
]
