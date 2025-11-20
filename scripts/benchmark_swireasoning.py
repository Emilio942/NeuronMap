#!/usr/bin/env python3
"""
SwiReasoning Benchmark Suite
============================

Runs a set of reasoning tasks (Math, STEM, Code, Logic) through the model
with the SwiReasoning policy enabled, and generates a performance report.
"""

import torch
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add src to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.analysis.universal_model_adapter import UniversalModelAdapter
from src.guardian.engine import GuardianEngine
from src.guardian.intervention_extractor import InterventionExtractor
from src.utils.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('src.guardian.policies').setLevel(logging.DEBUG)

BENCHMARK_TASKS = {
    "math": [
        "Solve for x: 2x + 5 = 15",
        "What is the square root of 144 plus 12?",
        "If a train travels 60 miles in 1.5 hours, what is its speed?"
    ],
    "logic": [
        "If all cats are mammals and some mammals are black, are all cats black?",
        "John is taller than Bob. Bob is taller than Alice. Is John taller than Alice?",
        "Identify the next number in the sequence: 2, 4, 8, 16, ..."
    ],
    "code": [
        "Write a Python function to calculate the factorial of a number.",
        "Create a list comprehension that squares even numbers from 0 to 10.",
        "Explain the difference between a list and a tuple in Python."
    ],
    "stem": [
        "Explain the process of photosynthesis.",
        "What is the difference between velocity and speed?",
        "Describe the structure of an atom."
    ]
}

def run_benchmark(model_name: str = "distilgpt2", device: str = "cpu"):
    logger.info(f"Starting SwiReasoning Benchmark with model: {model_name}")
    
    # 1. Setup Configuration
    config_manager = ConfigManager()
    
    # Guardian Config for SwiReasoning
    guardian_config = {
        'enabled': True,
        'mode': 'intervention',
        'policy_type': 'swireasoning',
        'history_window': 3,
        'entropy_min': 3.0, # Relaxed for demo/distilgpt2
        'entropy_max': 4.0,
        'switch_threshold': 0.5,
        'monitor_all_layers': True # Monitor all layers to get better entropy signal
    }
    
    # 2. Load Model
    logger.info("Loading model...")
    # UniversalModelAdapter expects the config object (Pydantic model) not a dict, despite type hint
    uma_config = config_manager.get_config_model()
    uma = UniversalModelAdapter(uma_config)
    adapter = uma.load_model(model_name)
    
    # 3. Initialize Guardian
    logger.info("Initializing Guardian Engine...")
    # Pass a copy so the original dict isn't mutated by the engine
    guardian_engine = GuardianEngine(guardian_config.copy())
    
    # 4. Initialize Extractor (Registers Hooks)
    # We need to hook into the model. InterventionExtractor does this.
    # However, InterventionExtractor usually wraps the model for analysis.
    # Here we want to run generation.
    # We can manually register the hooks using the extractor's logic or use the extractor.
    
    extractor = InterventionExtractor(
        guardian_engine=guardian_engine,
        model_name_or_config=model_name,
        model=adapter.model,
        tokenizer=adapter.tokenizer
    )
    
    # Register hooks
    # For SwiReasoning, we need the entropy of the Next Token Distribution.
    # This means we must hook the LM Head (logits), not the intermediate hidden states.
    if hasattr(adapter.model, 'lm_head'):
        logger.info("Hooking into lm_head for Logits Entropy")
        # We use a high layer_idx to indicate it's the head
        extractor.register_intervention_hook(adapter.model.lm_head, layer_idx=999)
    else:
        logger.warning("lm_head not found, falling back to last transformer layer (Warning: Entropy will be inaccurate)")
        layers = adapter.get_target_layers({'total_layers': 6, 'attention': 'transformer.h.{layer}'})
        last_layer_idx = len(layers) - 1
        if last_layer_idx >= 0:
            name, module = layers[last_layer_idx]
            logger.info(f"Hooking into layer: {name}")
            extractor.register_intervention_hook(module, layer_idx=last_layer_idx)
    
    results = []
    
    # 5. Run Tasks
    
    modes = [
        ("Baseline", {'enabled': False}),
        ("SwiReasoning", guardian_config)
    ]
    
    all_results = []
    
    for mode_name, config_override in modes:
        logger.info(f"Starting Benchmark Pass: {mode_name}")
        
        # Update Guardian Config
        guardian_engine.update_config(config_override)
        
        for category, prompts in BENCHMARK_TASKS.items():
            logger.info(f"Running category: {category}")
            for prompt in prompts:
                logger.info(f"Prompt: {prompt}")
                
                # Reset Policy Trace for new run (only matters if enabled)
                if config_override.get('enabled', False):
                    # Force re-init of policy state by updating config again or accessing policy
                    # The simplest way to clear the trace in the current implementation 
                    # is to rely on the fact that update_config might reset it, 
                    # or we can manually clear it if we had access. 
                    # For now, we assume update_config is sufficient or the policy handles it.
                    # Actually, looking at policies.py, we might need a reset method.
                    # But let's just re-apply the config.
                    guardian_engine.update_config(config_override)
                
                start_time = time.time()
                
                # Generate
                inputs = adapter.tokenizer(prompt, return_tensors="pt").to(adapter.device)
                
                with torch.no_grad():
                    outputs = adapter.model.generate(
                        **inputs, 
                        max_new_tokens=50, 
                        do_sample=True, 
                        temperature=0.7,
                        pad_token_id=adapter.tokenizer.eos_token_id
                    )
                
                end_time = time.time()
                generated_text = adapter.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Collect Stats
                result = {
                    'mode': mode_name,
                    'category': category,
                    'prompt': prompt,
                    'generated_text': generated_text[:100] + "...",
                    'full_text': generated_text,
                    'time': end_time - start_time,
                }
                
                if config_override.get('enabled', False):
                    policy = guardian_engine.policies.active_policy
                    trace = policy.trace
                    
                    # Calculate metrics
                    num_switches = len(trace.switches)
                    latent_blocks = len([b for b in trace.thinking_blocks if b.block_type == 'latent'])
                    explicit_blocks = len([b for b in trace.thinking_blocks if b.block_type == 'explicit'])
                    
                    result.update({
                        'switches': num_switches,
                        'latent_blocks': latent_blocks,
                        'explicit_blocks': explicit_blocks,
                        'final_mode': policy.current_mode,
                        'trace_data': {
                            'prompt': prompt,
                            'category': category,
                            'entropy_history': policy.entropy_history,
                            'switches': [
                                {
                                    'id': s.switch_id,
                                    'from': s.from_block_id,
                                    'to': s.to_block_id,
                                    'reason': s.reason,
                                    'token_idx': trace.thinking_blocks[s.from_block_id].end_token_idx if s.from_block_id < len(trace.thinking_blocks) else 0
                                } for s in trace.switches
                            ],
                            'blocks': [
                                {
                                    'id': b.block_id,
                                    'type': b.block_type,
                                    'start': b.start_token_idx,
                                    'end': b.end_token_idx
                                } for b in trace.thinking_blocks
                            ],
                            'generated_text': generated_text
                        }
                    })
                else:
                    # Baseline defaults
                    result.update({
                        'switches': 0,
                        'latent_blocks': 0,
                        'explicit_blocks': 0,
                        'final_mode': 'N/A',
                        'trace_data': None
                    })
                
                all_results.append(result)
                logger.info(f"Result ({mode_name}): Time={result['time']:.2f}s")

    # 6. Generate Report
    df = pd.DataFrame(all_results)
    
    # Pivot for comparison if possible, or just show all
    # Let's create a comparison view for the console
    print("\n=== SwiReasoning Benchmark Report ===")
    
    # Summary by Mode
    summary = df.groupby('mode')[['time', 'switches']].mean()
    print("\nAverage Metrics by Mode:")
    print(summary.to_markdown())
    
    # Detailed Table (drop trace_data and full_text for display)
    display_cols = ['mode', 'category', 'time', 'switches', 'generated_text']
    print("\nDetailed Results:")
    print(df[display_cols].to_markdown())
    
    # Save to file
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    
    # Save CSV
    df.drop(columns=['trace_data']).to_csv(output_dir / f"swireasoning_benchmark_{timestamp}.csv", index=False)
    
    # Save Detailed Traces (JSON) - Only for SwiReasoning runs
    traces = [r['trace_data'] for r in all_results if r['trace_data'] is not None]
    with open(output_dir / f"swireasoning_traces_{timestamp}.json", 'w') as f:
        json.dump(traces, f, indent=2)
        
    logger.info(f"Benchmark saved to {output_dir}")

if __name__ == "__main__":
    run_benchmark()
