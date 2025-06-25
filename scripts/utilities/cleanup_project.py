#!/usr/bin/env python3
"""
NeuronMap Project Cleanup Script

Dieses Script räumt das Projekt auf und organisiert Dateien in die richtige Struktur.
"""

import os
import shutil
import glob
from pathlib import Path
from datetime import datetime

def main():
    """Hauptfunktion für das Projekt-Cleanup."""
    
    base_dir = Path(__file__).parent.parent  # NeuronMap root directory
    os.chdir(base_dir)
    
    print("🧹 NeuronMap Projekt-Cleanup gestartet...")
    print(f"📂 Arbeitsverzeichnis: {base_dir}")
    
    # Erstelle Ordner falls sie nicht existieren
    create_directories()
    
    # Räume verschiedene Dateitypen auf
    cleanup_status_files()
    cleanup_validation_scripts()
    cleanup_debug_scripts()
    cleanup_demo_files()
    cleanup_data_files()
    cleanup_log_files()
    cleanup_test_files()
    
    # Entferne leere __pycache__ Ordner
    cleanup_pycache()
    
    print("✅ Projekt-Cleanup abgeschlossen!")
    print_directory_summary()

def create_directories():
    """Erstelle notwendige Ordnerstruktur."""
    directories = [
        "archive",
        "archive/status_reports", 
        "archive/validation_scripts",
        "archive/debug_scripts",
        "archive/temp_files",
        "scripts",
        "scripts/utilities",
        "demos",
        "logs",
        "data",
        "outputs",
        "checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def cleanup_status_files():
    """Räume Status-Report-Dateien auf."""
    print("📋 Räume Status-Reports auf...")
    
    status_patterns = [
        "*COMPLETE*.md",
        "*STATUS*.md", 
        "*PROGRESS*.md",
        "SECTION_*.md",
        "*ACCOMPLISHMENTS*.md",
        "*MISSION*.md",
        "*ENHANCEMENT*.md",
        "*PHASE*.md",
        "*PROJEKT*.md",
        "*ZEIT*.md"
    ]
    
    for pattern in status_patterns:
        for file in glob.glob(pattern):
            if os.path.isfile(file):
                shutil.move(file, "archive/status_reports/")
                print(f"  📁 {file} → archive/status_reports/")

def cleanup_validation_scripts():
    """Räume Validierungs-Scripts auf."""
    print("🔍 Räume Validierungs-Scripts auf...")
    
    validation_patterns = [
        "validate_*.py",
        "test_*.py", 
        "check_*.py",
        "simple_validation.py",
        "section_*.py"
    ]
    
    for pattern in validation_patterns:
        for file in glob.glob(pattern):
            if os.path.isfile(file):
                shutil.move(file, "archive/validation_scripts/")
                print(f"  🔍 {file} → archive/validation_scripts/")

def cleanup_debug_scripts():
    """Räume Debug-Scripts auf."""
    print("🐛 Räume Debug-Scripts auf...")
    
    debug_patterns = [
        "debug_*.py",
        "fix_*.py",
        "create_specialists.py",
        "migrate_*.py"
    ]
    
    for pattern in debug_patterns:
        for file in glob.glob(pattern):
            if os.path.isfile(file):
                shutil.move(file, "archive/debug_scripts/")
                print(f"  🐛 {file} → archive/debug_scripts/")

def cleanup_demo_files():
    """Räume Demo-Dateien auf."""
    print("🎯 Räume Demo-Dateien auf...")
    
    demo_patterns = [
        "demo_*.py",
        "demo_*.sh",
        "demo_*.txt"
    ]
    
    for pattern in demo_patterns:
        for file in glob.glob(pattern):
            if os.path.isfile(file):
                shutil.move(file, "demos/")
                print(f"  🎯 {file} → demos/")

def cleanup_data_files():
    """Räume Datendateien auf."""
    print("📊 Räume Datendateien auf...")
    
    data_patterns = [
        "*.csv",
        "*.jsonl", 
        "*.json",
        "*.db",
        "*results*.txt",
        "*questions*.txt"
    ]
    
    for pattern in data_patterns:
        for file in glob.glob(pattern):
            if os.path.isfile(file) and not file.startswith("pyproject"):
                shutil.move(file, "data/")
                print(f"  📊 {file} → data/")

def cleanup_log_files():
    """Räume Log-Dateien auf."""
    print("📝 Räume Log-Dateien auf...")
    
    log_patterns = [
        "*.log",
        "*_log.txt"
    ]
    
    for pattern in log_patterns:
        for file in glob.glob(pattern):
            if os.path.isfile(file):
                shutil.move(file, "logs/")
                print(f"  📝 {file} → logs/")

def cleanup_test_files():
    """Räume Test-Output-Dateien auf."""
    print("🧪 Räume Test-Dateien auf...")
    
    test_patterns = [
        "test_*.txt",
        "test_*.md",
        "*_test_*"
    ]
    
    for pattern in test_patterns:
        for file in glob.glob(pattern):
            if os.path.isfile(file):
                shutil.move(file, "archive/temp_files/")
                print(f"  🧪 {file} → archive/temp_files/")

def cleanup_pycache():
    """Entferne leere __pycache__ Ordner."""
    print("🗑️ Entferne leere __pycache__ Ordner...")
    
    for pycache_dir in glob.glob("**/__pycache__", recursive=True):
        try:
            if os.path.isdir(pycache_dir):
                # Prüfe ob Ordner leer ist
                if not os.listdir(pycache_dir):
                    os.rmdir(pycache_dir)
                    print(f"  🗑️ Entfernt: {pycache_dir}")
        except OSError:
            pass  # Ordner nicht leer oder andere Probleme

def print_directory_summary():
    """Zeige eine Zusammenfassung der Ordnerstruktur."""
    print("\n📁 Finale Ordnerstruktur:")
    
    important_dirs = [
        "src/",
        "docs/", 
        "scripts/",
        "demos/",
        "tests/",
        "data/",
        "logs/",
        "configs/",
        "archive/"
    ]
    
    for directory in important_dirs:
        if os.path.isdir(directory):
            file_count = len([f for f in os.listdir(directory) 
                            if os.path.isfile(os.path.join(directory, f))])
            dir_count = len([f for f in os.listdir(directory) 
                           if os.path.isdir(os.path.join(directory, f))])
            print(f"  📂 {directory:<15} ({file_count} Dateien, {dir_count} Ordner)")

if __name__ == "__main__":
    main()
