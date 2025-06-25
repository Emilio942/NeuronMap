# Data Generation Module API Reference

The data generation module handles question generation and synthetic data creation for neural network analysis.

## Overview

```{eval-rst}
.. automodule:: src.data_generation
   :members:
   :undoc-members:
   :show-inheritance:
```

## Core Classes

### QuestionGenerator

```{eval-rst}
.. autoclass:: src.data_generation.QuestionGenerator
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: generate_questions
   .. automethod:: generate_batch
   .. automethod:: validate_questions
```

### EnhancedQuestionGenerator

```{eval-rst}
.. autoclass:: src.data_generation.EnhancedQuestionGenerator
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Examples

### Basic Question Generation

```python
from src.data_generation import QuestionGenerator
from src.utils.config import get_config_manager

# Initialize generator
config = get_config_manager()
generator = QuestionGenerator(config=config)

# Generate questions
questions = generator.generate_questions(
    num_questions=100,
    categories=["science", "philosophy"],
    output_file="generated_questions.jsonl"
)

print(f"Generated {len(questions)} questions")
```

### Advanced Generation with Metadata

```python
from src.data_generation import EnhancedQuestionGenerator

generator = EnhancedQuestionGenerator()

# Generate with difficulty control
questions_with_metadata = generator.generate_with_difficulty(
    num_questions=50,
    difficulty_levels=["easy", "medium", "hard"],
    ensure_balance=True
)

# Analyze question quality
quality_report = generator.analyze_question_quality(questions_with_metadata)
print(f"Average quality score: {quality_report['average_score']}")
```

### Domain-Specific Generation

```python
from src.data_generation import DomainSpecificGenerator

# Science domain generator
science_generator = DomainSpecificGenerator(domain="science")

questions = science_generator.generate_domain_questions(
    subdomains=["physics", "chemistry", "biology"],
    complexity_range=("undergraduate", "graduate"),
    num_per_subdomain=20
)
```

## Configuration Options

### Generation Parameters

```python
generation_config = {
    'model_name': 'deepseek-r1:32b',
    'temperature': 0.7,
    'max_tokens': 150,
    'retry_attempts': 3,
    'batch_size': 20
}

generator = QuestionGenerator(config=generation_config)
```

### Quality Control

```python
quality_config = {
    'min_length': 10,
    'max_length': 200,
    'check_grammar': True,
    'filter_duplicates': True,
    'validate_semantic_coherence': True
}

generator.set_quality_config(quality_config)
```

## Function Reference

```{eval-rst}
.. autofunction:: src.data_generation.parse_questions_from_response
.. autofunction:: src.data_generation.validate_question_quality
.. autofunction:: src.data_generation.count_existing_questions
```

## Data Structures

### Question Metadata

```{eval-rst}
.. autoclass:: src.data_generation.QuestionMetadata
   :members:
   :undoc-members:
   :show-inheritance:
```

### Quality Metrics

```{eval-rst}
.. autoclass:: src.data_generation.QualityMetrics
   :members:
   :undoc-members:
   :show-inheritance:
```
