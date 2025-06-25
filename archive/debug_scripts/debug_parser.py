#!/usr/bin/env python3
"""Test script to debug argument parsing issues."""

import argparse
import sys

def test_parser():
    """Test the argument parser setup."""
    parser = argparse.ArgumentParser(description="Test Parser")
    parser.add_argument("--config", default="default")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    print("Setting up subparsers...")
    
    # Test basic parser
    test_parser = subparsers.add_parser("test", help="Test command")
    test_parser.add_argument("--input", help="Input file")
    print("✓ Test parser added")
    
    # Test domain parser (problematic one)
    domain_parser = subparsers.add_parser("domain", help="Domain analysis")
    domain_parser.add_argument("--analysis-type", required=True, choices=["code", "math"])
    domain_parser.add_argument("--model", required=True, help="Model name")
    # Missing set_defaults - let's see if this breaks things
    print("✓ Domain parser added")
    
    # Test ethics parser  
    ethics_parser = subparsers.add_parser("ethics", help="Ethics analysis")
    ethics_parser.add_argument("--model", required=True, help="Model name")
    ethics_parser.add_argument("--texts-file", required=True, help="Text file")
    print("✓ Ethics parser added")
    
    # Test conceptual parser
    conceptual_parser = subparsers.add_parser("conceptual", help="Conceptual analysis")
    conceptual_parser.add_argument("--analysis-type", required=True, choices=["concepts", "circuits"])
    conceptual_parser.add_argument("--model", required=True, help="Model name")
    print("✓ Conceptual parser added")
    
    print("All parsers added successfully!")
    
    # Test parsing help
    try:
        if len(sys.argv) == 1:
            sys.argv.append("--help")
        args = parser.parse_args()
        print(f"Parsed args: {args}")
    except SystemExit as e:
        print(f"SystemExit: {e.code}")
        if e.code == 0:
            print("Help displayed successfully")
        else:
            print("Error in parsing")

if __name__ == "__main__":
    test_parser()
