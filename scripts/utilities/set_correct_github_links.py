#!/usr/bin/env python3
"""
Script to set correct GitHub repository links for Emilio942/NeuronMap
"""

import os
import re
from pathlib import Path

def set_correct_github_links():
    """Set correct GitHub repository links for Emilio942/NeuronMap."""
    
    print("🔧 Setze korrekte GitHub-Links für Emilio942/NeuronMap...")
    
    correct_repo = "https://github.com/Emilio942/NeuronMap"
    correct_user = "Emilio942"
    
    # Patterns to replace
    patterns_to_replace = [
        (r'\[LOCAL_PROJECT\]', correct_repo),
        (r'# Lokales Projekt - kein git clone nötig', f'git clone {correct_repo}.git'),
        (r'https://github\.com/emilio/neuronmap', correct_repo),
        (r'https://github\.com/neuronmap/neuronmap', correct_repo),
        (r'https://github\.com/your-org/neuronmap', correct_repo),
        (r'https://github\.com/your-repo/neuronmap', correct_repo),
        (r'https://github\.com/username/neuronmap', correct_repo),
        (r'cd neuronmap', 'cd NeuronMap'),
        (r'cd NeuronMap\n', 'cd NeuronMap\n'),
    ]
    
    # Find all markdown files
    md_files = []
    for root, dirs, files in os.walk('.'):
        # Skip .git and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
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
                print(f"  ✅ Aktualisiert: {file_path}")
        
        except Exception as e:
            print(f"  ❌ Fehler bei {file_path}: {e}")
    
    print(f"\n🎉 {len(fixed_files)} Dateien mit korrekten GitHub-Links aktualisiert!")
    
    if fixed_files:
        print("\nAktualisierte Dateien:")
        for file in fixed_files:
            print(f"  - {file}")
    
    print(f"\n✅ Repository: {correct_repo}")
    print(f"✅ Benutzer: {correct_user}")

if __name__ == "__main__":
    set_correct_github_links()
