"""
Overthinking Detector
=====================

Analyzes SwiReasoning traces to detect inefficient reasoning patterns,
such as rapid mode switching (thrashing) or getting stuck in latent loops.
"""

from typing import Dict, List, Any
import numpy as np

class OverthinkingDetector:
    def __init__(self):
        self.thrashing_threshold = 3  # Max switches allowed in window
        self.window_size = 10         # Token window size
        self.max_latent_length = 50   # Max tokens in latent mode before warning

    def analyze_trace(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a trace for overthinking patterns.
        
        Args:
            trace_data: Dictionary containing 'switches', 'blocks', 'entropy_history'
            
        Returns:
            Dictionary with analysis results (flags, scores, warnings)
        """
        switches = trace_data.get('switches', [])
        blocks = trace_data.get('blocks', [])
        entropy = trace_data.get('entropy_history', [])
        
        issues = []
        score = 100.0 # Start with perfect score
        
        # 1. Detect Rapid Switching (Thrashing)
        if len(switches) >= self.thrashing_threshold:
            switch_indices = [s['token_idx'] for s in switches]
            for i in range(len(switch_indices) - self.thrashing_threshold + 1):
                window = switch_indices[i : i + self.thrashing_threshold]
                span = window[-1] - window[0]
                if span < self.window_size:
                    issues.append({
                        'type': 'thrashing',
                        'severity': 'high',
                        'message': f"Rapid switching detected: {self.thrashing_threshold} switches in {span} tokens at index {window[0]}",
                        'location': window[0]
                    })
                    score -= 25

        # 2. Detect Stuck in Latent Mode
        for block in blocks:
            if block['type'] == 'latent':
                end_idx = block.get('end')
                if end_idx is None or end_idx == 0: # Handle open block
                    end_idx = len(entropy)
                
                length = end_idx - block['start']
                    
                if length > self.max_latent_length:
                    issues.append({
                        'type': 'stuck_latent',
                        'severity': 'medium',
                        'message': f"Extended latent block detected ({length} tokens) starting at {block['start']}",
                        'location': block['start']
                    })
                    score -= 10

        # 3. Detect High Entropy Variance (Instability)
        if len(entropy) > 0:
            entropy_std = np.std(entropy)
            if entropy_std > 1.5: # Arbitrary threshold for now
                 issues.append({
                        'type': 'instability',
                        'severity': 'low',
                        'message': f"High entropy variance ({entropy_std:.2f}) indicates unstable confidence",
                        'location': 0
                    })
                 score -= 5

        # 4. Detect Repetition Loops
        text = trace_data.get('generated_text', '')
        if text:
            # Simple N-gram repetition check
            def has_repetition(s, n=15, threshold=3):
                # Check for repeated substrings of length n
                seen = {}
                for i in range(len(s) - n + 1):
                    gram = s[i:i+n]
                    seen[gram] = seen.get(gram, 0) + 1
                    if seen[gram] >= threshold:
                        return True
                return False
            
            if has_repetition(text):
                 issues.append({
                        'type': 'repetition_loop',
                        'severity': 'high',
                        'message': "Repetitive text generation detected",
                        'location': 0
                    })
                 score -= 30

        return {
            'is_overthinking': score < 80,
            'score': max(0.0, score),
            'status': 'critical' if score < 50 else 'warning' if score < 80 else 'healthy',
            'flags': issues, # Alias for frontend compatibility
            'issues': issues,
            'metrics': {
                'switch_density': len(switches) / len(entropy) if entropy else 0,
                'avg_block_length': np.mean([(b['end'] if b['end'] is not None else len(entropy)) - b['start'] for b in blocks]) if blocks else 0
            }
        }
