"""Unit tests for data generation modules."""

import unittest
import pytest
import tempfile
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.data_generation.question_generator import QuestionGenerator, QualityManager
except ImportError as e:
    print(f"Warning: Could not import question generator modules: {e}")


class TestQuestionGenerator(unittest.TestCase):
    """Test question generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = {
            'ollama_url': 'http://localhost:11434',
            'model': 'llama2',
            'num_questions': 5,
            'output_file': str(Path(self.temp_dir.name) / 'questions.json'),
            'categories': ['factual', 'reasoning']
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    @patch('requests.post')
    def test_question_generator_init(self, mock_post):
        """Test QuestionGenerator initialization."""
        try:
            # Mock successful Ollama connection
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'response': 'Test question: What is artificial intelligence?'
            }
            mock_post.return_value = mock_response
            
            generator = QuestionGenerator(self.config)
            self.assertEqual(generator.config['model'], 'llama2')
            self.assertEqual(generator.config['num_questions'], 5)
            
        except ImportError:
            self.skipTest("QuestionGenerator not available")
    
    @patch('requests.post')
    def test_generate_single_question(self, mock_post):
        """Test generating a single question."""
        try:
            # Mock Ollama response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'response': 'What is the capital of France?'
            }
            mock_post.return_value = mock_response
            
            generator = QuestionGenerator(self.config)
            question = generator._generate_single_question('factual')
            
            self.assertIsInstance(question, dict)
            self.assertIn('text', question)
            self.assertIn('category', question)
            self.assertEqual(question['category'], 'factual')
            
        except ImportError:
            self.skipTest("QuestionGenerator not available")
    
    def test_validate_question_quality(self):
        """Test question quality validation."""
        try:
            generator = QuestionGenerator(self.config)
            
            # Test valid question
            valid_question = "What is the largest planet in our solar system?"
            self.assertTrue(generator._is_valid_question(valid_question))
            
            # Test invalid questions
            self.assertFalse(generator._is_valid_question(""))  # Empty
            self.assertFalse(generator._is_valid_question("Hi"))  # Too short
            self.assertFalse(generator._is_valid_question("This is not a question"))  # No question mark
            
        except ImportError:
            self.skipTest("QuestionGenerator not available")
    
    @patch('requests.post')
    def test_generate_questions_batch(self, mock_post):
        """Test batch question generation."""
        try:
            # Mock Ollama responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'response': 'What is machine learning?'
            }
            mock_post.return_value = mock_response
            
            generator = QuestionGenerator(self.config)
            questions = generator.generate_questions()
            
            self.assertIsInstance(questions, list)
            self.assertGreater(len(questions), 0)
            
            # Check question structure
            for question in questions:
                self.assertIn('text', question)
                self.assertIn('category', question)
                self.assertIn('timestamp', question)
                
        except ImportError:
            self.skipTest("QuestionGenerator not available")
    
    def test_save_questions(self):
        """Test saving questions to file."""
        try:
            test_questions = [
                {'text': 'What is AI?', 'category': 'factual', 'timestamp': '2024-01-01'},
                {'text': 'How do neural networks learn?', 'category': 'reasoning', 'timestamp': '2024-01-01'}
            ]
            
            generator = QuestionGenerator(self.config)
            generator._save_questions(test_questions, self.config['output_file'])
            
            # Verify file was created and contains correct data
            self.assertTrue(os.path.exists(self.config['output_file']))
            
            with open(self.config['output_file'], 'r') as f:
                saved_data = json.load(f)
            
            self.assertEqual(len(saved_data['questions']), 2)
            self.assertEqual(saved_data['questions'][0]['text'], 'What is AI?')
            
        except ImportError:
            self.skipTest("QuestionGenerator not available")


class TestQualityManager(unittest.TestCase):
    """Test question quality management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_questions = [
            {'text': 'What is artificial intelligence?', 'category': 'factual'},
            {'text': 'What is artificial intelligence?', 'category': 'factual'},  # Duplicate
            {'text': 'How do neural networks work?', 'category': 'reasoning'},
            {'text': 'Short', 'category': 'factual'},  # Too short
            {'text': '', 'category': 'factual'},  # Empty
            {'text': 'This is a statement without question mark', 'category': 'factual'},  # No question
        ]
    
    def test_duplicate_detection(self):
        """Test duplicate question detection."""
        try:
            quality_manager = QualityManager()
            
            duplicates = quality_manager.detect_duplicates(self.test_questions)
            self.assertGreater(len(duplicates), 0)
            
            # Should detect the duplicate "What is artificial intelligence?"
            duplicate_texts = [q['text'] for q in duplicates]
            self.assertIn('What is artificial intelligence?', duplicate_texts)
            
        except ImportError:
            self.skipTest("QualityManager not available")
    
    def test_similarity_detection(self):
        """Test semantic similarity detection."""
        try:
            quality_manager = QualityManager()
            
            similar_questions = [
                {'text': 'What is AI?', 'category': 'factual'},
                {'text': 'What is artificial intelligence?', 'category': 'factual'},
                {'text': 'How does machine learning work?', 'category': 'reasoning'}
            ]
            
            similar_pairs = quality_manager.find_similar_questions(similar_questions)
            self.assertIsInstance(similar_pairs, list)
            
        except ImportError:
            self.skipTest("QualityManager not available")
    
    def test_question_validation(self):
        """Test individual question validation."""
        try:
            quality_manager = QualityManager()
            
            # Test valid question
            valid_question = {'text': 'What is the capital of France?', 'category': 'factual'}
            self.assertTrue(quality_manager.is_valid_question(valid_question))
            
            # Test invalid questions
            invalid_questions = [
                {'text': '', 'category': 'factual'},  # Empty
                {'text': 'Hi', 'category': 'factual'},  # Too short
                {'text': 'This is not a question', 'category': 'factual'},  # No question mark
            ]
            
            for invalid_q in invalid_questions:
                self.assertFalse(quality_manager.is_valid_question(invalid_q))
                
        except ImportError:
            self.skipTest("QualityManager not available")
    
    def test_quality_scoring(self):
        """Test question quality scoring."""
        try:
            quality_manager = QualityManager()
            
            high_quality = {'text': 'What are the key differences between supervised and unsupervised machine learning algorithms?', 'category': 'reasoning'}
            low_quality = {'text': 'What is AI?', 'category': 'factual'}
            
            high_score = quality_manager.calculate_quality_score(high_quality)
            low_score = quality_manager.calculate_quality_score(low_quality)
            
            self.assertGreater(high_score, low_score)
            self.assertIsInstance(high_score, (int, float))
            self.assertIsInstance(low_score, (int, float))
            
        except ImportError:
            self.skipTest("QualityManager not available")


class TestQuestionCategories(unittest.TestCase):
    """Test question categorization functionality."""
    
    def test_category_validation(self):
        """Test that question categories are properly validated."""
        try:
            from src.data_generation.question_generator import VALID_CATEGORIES
            
            expected_categories = ['factual', 'reasoning', 'creative', 'ethical']
            
            for category in expected_categories:
                self.assertIn(category, VALID_CATEGORIES)
                
        except ImportError:
            self.skipTest("Category validation not available")
    
    def test_category_specific_generation(self):
        """Test generation of category-specific questions."""
        try:
            config = {
                'ollama_url': 'http://localhost:11434',
                'model': 'llama2',
                'categories': ['factual']
            }
            
            with patch('requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    'response': 'What is the capital of Spain?'
                }
                mock_post.return_value = mock_response
                
                generator = QuestionGenerator(config)
                question = generator._generate_single_question('factual')
                
                self.assertEqual(question['category'], 'factual')
                
        except ImportError:
            self.skipTest("QuestionGenerator not available")


class MockOllamaTests(unittest.TestCase):
    """Test question generation with mocked Ollama service."""
    
    def test_ollama_connection_error(self):
        """Test handling of Ollama connection errors."""
        try:
            config = {
                'ollama_url': 'http://localhost:11434',
                'model': 'llama2',
                'num_questions': 5
            }
            
            with patch('requests.post') as mock_post:
                # Simulate connection error
                mock_post.side_effect = Exception("Connection refused")
                
                with self.assertRaises(Exception):
                    generator = QuestionGenerator(config)
                    generator._generate_single_question('factual')
                    
        except ImportError:
            self.skipTest("QuestionGenerator not available")
    
    def test_invalid_response_handling(self):
        """Test handling of invalid Ollama responses."""
        try:
            config = {
                'ollama_url': 'http://localhost:11434',
                'model': 'llama2'
            }
            
            with patch('requests.post') as mock_post:
                # Simulate invalid response
                mock_response = Mock()
                mock_response.status_code = 500
                mock_response.json.return_value = {'error': 'Model not found'}
                mock_post.return_value = mock_response
                
                generator = QuestionGenerator(config)
                result = generator._generate_single_question('factual')
                
                # Should handle error gracefully
                self.assertIsNone(result)
                
        except ImportError:
            self.skipTest("QuestionGenerator not available")


if __name__ == "__main__":
    unittest.main(verbosity=2)