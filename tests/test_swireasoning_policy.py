import pytest
from src.guardian.policies import SwiReasoningPolicy
from src.guardian.swireasoning import ThinkingBlock, ReasoningSwitch

def test_swireasoning_policy_initialization():
    config = {'policy_type': 'swireasoning'}
    policy = SwiReasoningPolicy(config)
    assert policy.current_mode == "explicit"
    assert len(policy.trace.thinking_blocks) == 1
    assert policy.trace.thinking_blocks[0].block_type == "explicit"

def test_switch_to_latent():
    config = {
        'policy_type': 'swireasoning',
        'entropy_min': 0.5,
        'history_window': 3
    }
    policy = SwiReasoningPolicy(config)
    
    # Feed high confidence (low entropy) stable values
    # Need enough values to fill window and have low slope
    # 1. 0.4
    policy.decide({'entropy': 0.4})
    # 2. 0.4
    policy.decide({'entropy': 0.4})
    # 3. 0.4 -> Slope 0, Entropy 0.4 < 0.5 -> Switch
    action = policy.decide({'entropy': 0.4})
        
    # Should have switched to latent
    assert policy.current_mode == "latent"
    assert action['action'] == 'enter_latent_mode'
    assert len(policy.trace.switches) == 1
    assert policy.trace.switches[0].reason == "high_confidence"

def test_switch_back_to_explicit():
    config = {
        'policy_type': 'swireasoning',
        'entropy_max': 1.5,
        'switch_threshold': 0.1,
        'history_window': 3
    }
    policy = SwiReasoningPolicy(config)
    
    # Force into latent mode manually for testing transition
    policy.current_mode = "latent"
    policy.current_block.block_type = "latent"
    
    # Feed increasing entropy (uncertainty rise)
    # 1. 1.0
    policy.decide({'entropy': 1.0})
    # 2. 1.2
    policy.decide({'entropy': 1.2})
    # 3. 1.6 -> Entropy > 1.5 -> Switch
    action = policy.decide({'entropy': 1.6})
        
    # Should have switched to explicit
    assert policy.current_mode == "explicit"
    assert action['action'] == 'exit_latent_mode'
    assert len(policy.trace.switches) >= 1
    assert policy.trace.switches[-1].reason == "uncertainty_rise"
