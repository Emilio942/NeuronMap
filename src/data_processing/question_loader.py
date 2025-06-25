"""
Question Loader Module for NeuronMap
===================================

This module handles loading and preprocessing of questions from various file formats.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class QuestionLoader:
    """Handles loading questions from various file formats."""

    def __init__(self, config):
        self.config = config
        self.input_file = Path(config.data.input_file)

    def load_questions(self) -> List[str]:
        """Load questions from the configured input file."""
        if not self.input_file.exists():
            logger.error(f"Input file {self.input_file} not found")
            return []

        try:
            if self.input_file.suffix.lower() == '.jsonl':
                return self._load_jsonl()
            elif self.input_file.suffix.lower() == '.json':
                return self._load_json()
            elif self.input_file.suffix.lower() == '.csv':
                return self._load_csv()
            elif self.input_file.suffix.lower() == '.txt':
                return self._load_txt()
            else:
                logger.warning(f"Unknown file format {self.input_file.suffix}. Trying as text file.")
                return self._load_txt()

        except Exception as e:
            logger.error(f"Failed to load questions from {self.input_file}: {e}")
            return []

    def _load_jsonl(self) -> List[str]:
        """Load questions from JSONL file."""
        questions = []

        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        # Try common question field names
                        question = (data.get('question') or
                                  data.get('text') or
                                  data.get('prompt') or
                                  data.get('input'))
                        if question:
                            questions.append(str(question))
                        else:
                            logger.warning(f"Line {line_num}: No question field found in JSON object")
                    elif isinstance(data, str):
                        questions.append(data)
                    else:
                        logger.warning(f"Line {line_num}: Unexpected data type {type(data)}")

                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Failed to parse JSON: {e}")

        logger.info(f"Loaded {len(questions)} questions from JSONL file")
        return questions

    def _load_json(self) -> List[str]:
        """Load questions from JSON file."""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        questions = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    questions.append(item)
                elif isinstance(item, dict):
                    question = (item.get('question') or
                              item.get('text') or
                              item.get('prompt') or
                              item.get('input'))
                    if question:
                        questions.append(str(question))
        elif isinstance(data, dict):
            # Single question or questions under a key
            if 'questions' in data:
                questions = [str(q) for q in data['questions'] if q]
            else:
                question = (data.get('question') or
                          data.get('text') or
                          data.get('prompt') or
                          data.get('input'))
                if question:
                    questions.append(str(question))

        logger.info(f"Loaded {len(questions)} questions from JSON file")
        return questions

    def _load_csv(self) -> List[str]:
        """Load questions from CSV file."""
        df = pd.read_csv(self.input_file)

        # Try common column names for questions
        question_columns = ['question', 'text', 'prompt', 'input', 'query']
        question_col = None

        for col in question_columns:
            if col in df.columns:
                question_col = col
                break

        if question_col is None:
            # Use the first column
            question_col = df.columns[0]
            logger.warning(f"No standard question column found. Using '{question_col}'")

        questions = df[question_col].dropna().astype(str).tolist()

        logger.info(f"Loaded {len(questions)} questions from CSV file (column: {question_col})")
        return questions

    def _load_txt(self) -> List[str]:
        """Load questions from text file (one per line)."""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(questions)} questions from text file")
        return questions

    def validate_questions(self, questions: List[str]) -> List[str]:
        """Validate and filter questions."""
        valid_questions = []

        for i, question in enumerate(questions):
            # Basic validation
            if not question or not question.strip():
                logger.warning(f"Question {i+1}: Empty question, skipping")
                continue

            # Length check
            if len(question.strip()) < 3:
                logger.warning(f"Question {i+1}: Too short ({len(question)} chars), skipping")
                continue

            # Max length check (to prevent memory issues)
            max_length = getattr(self.config.model, 'max_length', 512)
            if len(question) > max_length * 4:  # Rough estimate for token limit
                logger.warning(f"Question {i+1}: Too long ({len(question)} chars), truncating")
                question = question[:max_length * 4]

            valid_questions.append(question.strip())

        logger.info(f"Validated {len(valid_questions)} out of {len(questions)} questions")
        return valid_questions

    def save_sample_questions(self, output_path: str = None) -> None:
        """Save a sample questions file for testing."""
        if output_path is None:
            output_path = self.config.data.raw_dir + "/sample_questions.jsonl"

        sample_questions = [
            {"question": "What is the capital of France?"},
            {"question": "How does photosynthesis work?"},
            {"question": "What are the benefits of machine learning?"},
            {"question": "Explain the theory of relativity."},
            {"question": "What is the meaning of life?"},
            {"question": "How do neural networks learn?"},
            {"question": "What is quantum computing?"},
            {"question": "Describe the water cycle."},
            {"question": "What causes climate change?"},
            {"question": "How does the human brain work?"}
        ]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for question_data in sample_questions:
                f.write(json.dumps(question_data) + '\n')

        logger.info(f"Sample questions saved to {output_path}")


def create_sample_data(config):
    """Create sample data files for testing."""
    loader = QuestionLoader(config)
    loader.save_sample_questions()

    # Also create the questions file expected by the current system
    sample_questions_file = Path("generated_questions.jsonl")
    if not sample_questions_file.exists():
        loader.save_sample_questions(str(sample_questions_file))
        print(f"Created sample questions file: {sample_questions_file}")


if __name__ == "__main__":
    # Test the question loader
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config_manager import get_config

    config = get_config()
    create_sample_data(config)
