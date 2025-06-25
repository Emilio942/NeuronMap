#!/usr/bin/env python3
"""
Script to fix all incorrect GitHub links in NeuronMap documentation
"""

import os
import re
from pathlib import Path

def fix_github_links():
    """Fix all incorrect GitHub repository links in documentation."""
    
    print("🔧 Korrigiere GitHub-Links in der Dokumentation...")
    
    # Patterns to replace
    patterns_to_replace = [
        (r'https://github\.com/emilio/neuronmap', '[LOCAL_PROJECT]'),
        (r'https://github\.com/neuronmap/neuronmap', '[LOCAL_PROJECT]'),
        (r'https://github\.com/your-org/neuronmap', '[LOCAL_PROJECT]'),
        (r'https://github\.com/your-repo/neuronmap', '[LOCAL_PROJECT]'),
        (r'https://github\.com/username/neuronmap', '[LOCAL_PROJECT]'),
        (r'git clone https://github\.com/[^/]+/neuronmap\.git', '# Lokales Projekt - kein git clone nötig'),
        (r'git clone -b develop https://github\.com/[^/]+/neuronmap\.git', '# Lokales Projekt - kein git clone nötig'),
    ]
    
    # Find all markdown files
    md_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    
    fixed_files = []
    
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply all replacements
            for pattern, replacement in patterns_to_replace:
                content = re.sub(pattern, replacement, content)
            
            # Only write if content changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files.append(file_path)
                print(f"  ✅ Korrigiert: {file_path}")
        
        except Exception as e:
            print(f"  ❌ Fehler bei {file_path}: {e}")
    
    print(f"\n🎉 {len(fixed_files)} Dateien korrigiert!")
    
    if fixed_files:
        print("\nKorrigierte Dateien:")
        for file in fixed_files:
            print(f"  - {file}")

if __name__ == "__main__":
    fix_github_links()
