#!/usr/bin/env python3
"""
Quick fix for activation hook issues in multi-model support.
This script patches the hook functions to handle tuple outputs correctly.
"""

import re
import sys
from pathlib import Path

def fix_activation_hooks():
    """Fix activation hook functions to handle tuple outputs."""
    
    file_path = Path(__file__).parent / "src" / "utils" / "multi_model_support.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to find the problematic hook function
    old_hook_pattern = r'def capture_activation\(name\):\s+def hook\(module, input, output\):\s+captured_activations\[name\] = output\.detach\(\)\.cpu\(\)\s+return hook'
    
    # New hook function
    new_hook_function = '''def capture_activation(name):
            def hook(module, input, output):
                # Handle tuple outputs (some modules return tuples)
                if isinstance(output, tuple):
                    output = output[0]  # Take the first element (usually the hidden states)
                elif hasattr(output, 'last_hidden_state'):
                    output = output.last_hidden_state
                captured_activations[name] = output.detach().cpu()
            return hook'''
    
    # Replace all occurrences
    content = re.sub(old_hook_pattern, new_hook_function, content, flags=re.MULTILINE | re.DOTALL)
    
    # Also handle direct inline hook functions that might cause issues
    content = content.replace(
        'captured_activations[name] = output.detach().cpu()',
        '''# Handle tuple outputs
                if isinstance(output, tuple):
                    output = output[0]
                elif hasattr(output, 'last_hidden_state'):
                    output = output.last_hidden_state
                captured_activations[name] = output.detach().cpu()'''
    )
    
    # Write the file back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Fixed activation hook functions")

if __name__ == "__main__":
    fix_activation_hooks()
