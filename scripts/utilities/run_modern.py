#!/usr/bin/env python3
"""
Modern Run Script - Enhanced Activation Extraction
=================================================

Migrated from legacy run.py with enhancements:
- Modern modular architecture
- Enhanced error handling and logging
- Configuration system integration
- Flexible model and layer support
- Batch processing improvements
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.analysis.activation_extractor import ActivationExtractor
from src.utils.config_manager import ConfigManager

def main():
    """Enhanced main function with CLI support."""
    parser = argparse.ArgumentParser(description="NeuronMap Activation Extraction")
    parser.add_argument("--model", default="distilgpt2", help="Model name to use")
    parser.add_argument("--layer", default="transformer.h.5.mlp.c_proj", help="Target layer name")
    parser.add_argument("--questions", default="generated_questions.jsonl", help="Questions file")
    parser.add_argument("--output", default="activation_results.csv", help="Output file")
    parser.add_argument("--device", default="auto", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--print-layers", action="store_true", help="Print available layers and exit")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = ActivationExtractor(
        model_name_or_config=args.model,
        target_layer=args.layer,
        device=args.device
    )
    
    # Load model
    extractor.load_model_and_tokenizer()
    
    if args.print_layers:
        extractor.print_model_layers()
        return
    
    # Load questions and extract activations
    questions = extractor.load_questions(args.questions)
    if questions:
        results = extractor.extract_activations_batch(questions)
        extractor.save_results(results, args.output)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
