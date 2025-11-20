import unittest
from src.analysis.overthinking import OverthinkingDetector

class TestOverthinkingDetector(unittest.TestCase):
    def setUp(self):
        self.detector = OverthinkingDetector()

    def test_clean_trace(self):
        """Test a trace with no issues."""
        trace = {
            'switches': [{'token_idx': 10}, {'token_idx': 50}],
            'blocks': [
                {'type': 'explicit', 'start': 0, 'end': 10},
                {'type': 'latent', 'start': 10, 'end': 50},
                {'type': 'explicit', 'start': 50, 'end': 60}
            ],
            'entropy_history': [0.1] * 60
        }
        result = self.detector.analyze_trace(trace)
        self.assertFalse(result['is_overthinking'])
        self.assertEqual(result['score'], 100.0)
        self.assertEqual(len(result['issues']), 0)

    def test_thrashing(self):
        """Test detection of rapid switching."""
        # 3 switches in 5 tokens (indices 10, 12, 14)
        trace = {
            'switches': [{'token_idx': 10}, {'token_idx': 12}, {'token_idx': 14}],
            'blocks': [],
            'entropy_history': [0.1] * 20
        }
        result = self.detector.analyze_trace(trace)
        self.assertTrue(result['is_overthinking']) # Score < 80
        self.assertTrue(any(i['type'] == 'thrashing' for i in result['issues']))

    def test_stuck_latent(self):
        """Test detection of being stuck in latent mode."""
        # Latent block of length 60 (limit is 50)
        trace = {
            'switches': [],
            'blocks': [{'type': 'latent', 'start': 0, 'end': 60}],
            'entropy_history': [0.1] * 60
        }
        result = self.detector.analyze_trace(trace)
        # Score starts at 100, -10 for stuck latent = 90. Not "overthinking" (<80) but has issues.
        self.assertEqual(result['score'], 90.0)
        self.assertTrue(any(i['type'] == 'stuck_latent' for i in result['issues']))

if __name__ == '__main__':
    unittest.main()
