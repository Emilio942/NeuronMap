"""
SwiReasoning Data Models
========================

Data structures for tracking and managing the Switch-Thinking process
within the Guardian framework.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import time

@dataclass
class ThinkingBlock:
    """Represents a block of reasoning (either latent or explicit)."""
    block_id: int
    block_type: Literal["latent", "explicit"]
    start_token_idx: int
    end_token_idx: Optional[int] = None
    confidence_score: float = 0.0
    entropy_values: List[float] = field(default_factory=list)
    token_efficiency: float = 0.0
    
    @property
    def duration(self) -> int:
        if self.end_token_idx is None:
            return 0
        return self.end_token_idx - self.start_token_idx

@dataclass
class ReasoningSwitch:
    """Represents a switch event between reasoning modes."""
    switch_id: int
    from_block_id: int
    to_block_id: int
    timestamp: float = field(default_factory=time.time)
    switch_confidence: float = 0.0
    reason: str = "entropy_threshold"  # e.g., "entropy_threshold", "max_length", "manual"

@dataclass
class SwiReasoningTrace:
    """Complete trace of a reasoning session."""
    problem_id: str
    thinking_blocks: List[ThinkingBlock] = field(default_factory=list)
    switches: List[ReasoningSwitch] = field(default_factory=list)
    total_tokens: int = 0
    final_solution: str = ""
    is_correct: Optional[bool] = None
    
    def add_block(self, block: ThinkingBlock):
        self.thinking_blocks.append(block)
        
    def add_switch(self, switch: ReasoningSwitch):
        self.switches.append(switch)
