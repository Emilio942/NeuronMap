#!/usr/bin/env python3
"""
Project Charter & Technical Specifications Validation
====================================================

This script validates that the project charter and technical specifications
meet all requirements specified in aufgabenliste.md.
"""

import os
import sys
from pathlib import Path
import re

def validate_project_charter():
    """Validate project charter requirements."""
    print("🔍 Validating Project Charter...")
    
    charter_path = Path("docs/project_charter.md")
    if not charter_path.exists():
        print("❌ Project charter file missing")
        return False
    
    with open(charter_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    requirements = {
        "15+ transformer models": [
            "15+ transformer models",
            "Support for **15+ transformer models**"
        ],
        "25+ specialized commands": [
            "25+ specialized commands",
            "Command-line interface with 25+ specialized commands"
        ],
        "<100ms inference latency": [
            "<100ms inference latency", 
            "Real-time activation extraction with <100ms latency"
        ],
        "70B parameters support": [
            "70B parameters",
            "Support for models up to 70B parameters"
        ],
        ">95% code coverage": [
            ">95% for all core modules",
            "95% for all core modules"
        ],
        "Stakeholder analysis": [
            "Target User Personas & Stakeholder Analysis",
            "Stakeholder Analysis"
        ],
        "Risk assessment": [
            "Comprehensive Risk Assessment",
            "Risk Assessment & Mitigation"
        ],
        "Resource requirements": [
            "Resource Requirements",
            "Development Resources"
        ],
        "Timeline estimation": [
            "Timeline & Milestones",
            "Phase 1:", "Phase 2:", "Phase 3:", "Phase 4:"
        ]
    }
    
    errors = []
    for requirement, patterns in requirements.items():
        found = any(pattern in content for pattern in patterns)
        if found:
            print(f"✅ {requirement}: Found")
        else:
            print(f"❌ {requirement}: Missing")
            errors.append(requirement)
    
    return len(errors) == 0

def validate_technical_specs():
    """Validate technical specifications requirements."""
    print("\n🔍 Validating Technical Specifications...")
    
    specs_path = Path("docs/technical_specs.md")
    if not specs_path.exists():
        print("❌ Technical specifications file missing")
        return False
    
    with open(specs_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    requirements = {
        "Detailed architecture diagrams": [
            "Architecture Overview",
            "System Architecture Diagram"
        ],
        "API specifications": [
            "API Specifications",
            "Python API Design",
            "REST API Endpoints"
        ],
        "Performance requirements": [
            "Performance Requirements",
            "Performance Benchmarks",
            "Latency Benchmarks"
        ],
        "Integration guidelines": [
            "Integration Guidelines",
            "Framework Integration"
        ],
        "25+ CLI commands specified": [
            "25+ specialized commands",
            "Command-Line Interface Specifications"
        ],
        "Memory efficiency specs": [
            "Memory Optimization",
            "Memory Requirements",
            "memory-efficient"
        ],
        "Cross-platform compatibility": [
            "Cross-platform compatibility",
            "Cloud Platform Integration"
        ]
    }
    
    errors = []
    for requirement, patterns in requirements.items():
        found = any(pattern in content for pattern in patterns)
        if found:
            print(f"✅ {requirement}: Found")
        else:
            print(f"❌ {requirement}: Missing")
            errors.append(requirement)
    
    return len(errors) == 0

def validate_cli_commands():
    """Validate that 25+ CLI commands are specified."""
    print("\n🔍 Validating CLI Commands Specification...")
    
    specs_path = Path("docs/technical_specs.md")
    with open(specs_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count CLI commands
    command_patterns = [
        r'neuronmap\s+\w+\s+\w+',  # neuronmap category command
        r'neuronmap\s+\w+'         # neuronmap command
    ]
    
    commands = set()
    for pattern in command_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            # Clean up the command
            cmd = match.strip().replace('neuronmap ', '')
            if cmd and not cmd.startswith('<') and not cmd.startswith('['):
                commands.add(cmd)
    
    print(f"Found {len(commands)} unique CLI commands:")
    for cmd in sorted(commands):
        print(f"  • neuronmap {cmd}")
    
    if len(commands) >= 25:
        print(f"✅ CLI Commands: {len(commands)} commands found (≥25 required)")
        return True
    else:
        print(f"❌ CLI Commands: Only {len(commands)} commands found (<25 required)")
        return False

def validate_documentation_structure():
    """Validate overall documentation structure."""
    print("\n🔍 Validating Documentation Structure...")
    
    required_files = [
        "docs/project_charter.md",
        "docs/technical_specs.md",
        "README.md",
        "README_NEW.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}: Exists")
        else:
            print(f"❌ {file_path}: Missing")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def main():
    """Main validation function."""
    print("🎯 Project Charter & Technical Specifications Validation")
    print("=" * 60)
    
    os.chdir(Path(__file__).parent)
    
    results = []
    
    # Run all validations
    results.append(validate_project_charter())
    results.append(validate_technical_specs())
    results.append(validate_cli_commands())
    results.append(validate_documentation_structure())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = all(results)
    
    validation_names = [
        "Project Charter",
        "Technical Specifications", 
        "CLI Commands",
        "Documentation Structure"
    ]
    
    for i, (name, passed) in enumerate(zip(validation_names, results)):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
    
    print("")
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Project charter and technical specifications are complete")
        print("✅ All aufgabenliste.md requirements satisfied")
        print("")
        print("Ready to proceed with implementation!")
        return 0
    else:
        failed_count = len([r for r in results if not r])
        print(f"❌ {failed_count} VALIDATIONS FAILED")
        print("🔧 Please address the issues above before proceeding")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
