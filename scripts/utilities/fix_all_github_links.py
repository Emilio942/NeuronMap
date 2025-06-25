#!/usr/bin/env python3
"""
Comprehensive script to fix ALL GitHub links in NeuronMap documentation
"""

import os
import re
from pathlib import Path

def fix_all_github_links():
    """Fix ALL GitHub repository links comprehensively."""
    
    print("🔧 Comprehensive GitHub-Links Korrektur...")
    print("Suche nach ALLEN falschen GitHub-Links...")
    
    correct_repo = "https://github.com/Emilio942/NeuronMap"
    correct_user = "Emilio942"
    
    # All possible patterns to replace
    patterns_to_replace = [
        # Direct replacements
        (r'https://github\.com/your-username/NeuronMap', correct_repo),
        (r'https://github\.com/emilio/neuronmap', correct_repo),
        (r'https://github\.com/neuronmap/neuronmap', correct_repo),
        (r'https://github\.com/your-org/neuronmap', correct_repo),
        (r'https://github\.com/your-repo/neuronmap', correct_repo),
        (r'https://github\.com/username/neuronmap', correct_repo),
        (r'https://github\.com/your-org/NeuronMap', correct_repo),
        (r'https://github\.com/your-repo/NeuronMap', correct_repo),
        (r'https://github\.com/username/NeuronMap', correct_repo),
        
        # Git clone patterns  
        (r'git clone https://github\.com/your-username/NeuronMap\.git', f'git clone {correct_repo}.git'),
        (r'git clone https://github\.com/emilio/neuronmap\.git', f'git clone {correct_repo}.git'),
        (r'git clone https://github\.com/neuronmap/neuronmap\.git', f'git clone {correct_repo}.git'),
        (r'git clone https://github\.com/your-org/neuronmap\.git', f'git clone {correct_repo}.git'),
        (r'git clone https://github\.com/your-repo/neuronmap\.git', f'git clone {correct_repo}.git'),
        (r'git clone https://github\.com/username/neuronmap\.git', f'git clone {correct_repo}.git'),
        (r'git clone https://github\.com/your-org/NeuronMap\.git', f'git clone {correct_repo}.git'),
        (r'git clone https://github\.com/your-repo/NeuronMap\.git', f'git clone {correct_repo}.git'),
        (r'git clone https://github\.com/username/NeuronMap\.git', f'git clone {correct_repo}.git'),
        
        # Local project placeholders
        (r'\[LOCAL_PROJECT\]', correct_repo),
        (r'# Lokales Projekt - kein git clone nötig', f'git clone {correct_repo}.git'),
        
        # Directory changes
        (r'cd neuronmap', 'cd NeuronMap'),
    ]
    
    # Find all markdown files
    md_files = []
    for root, dirs, files in os.walk('.'):
        # Skip .git and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    
    print(f"Gefunden: {len(md_files)} Markdown-Dateien")
    
    fixed_files = []
    total_replacements = 0
    
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            file_replacements = 0
            
            # Apply all replacements
            for pattern, replacement in patterns_to_replace:
                matches = re.findall(pattern, content)
                if matches:
                    file_replacements += len(matches)
                content = re.sub(pattern, replacement, content)
            
            # Only write if content changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files.append((file_path, file_replacements))
                total_replacements += file_replacements
                print(f"  ✅ {file_path} ({file_replacements} Korrekturen)")
        
        except Exception as e:
            print(f"  ❌ Fehler bei {file_path}: {e}")
    
    print(f"\n🎉 Zusammenfassung:")
    print(f"  📄 {len(fixed_files)} Dateien korrigiert")
    print(f"  🔗 {total_replacements} Links korrigiert")
    print(f"  ✅ Alle Links zeigen jetzt auf: {correct_repo}")
    
    if fixed_files:
        print("\n📋 Detaillierte Korrekturen:")
        for file_path, count in fixed_files:
            print(f"  - {file_path}: {count} Links")

if __name__ == "__main__":
    fix_all_github_links()
