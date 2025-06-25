#!/usr/bin/env python3
"""
Debug what's available in domain_specialists module
"""
import sys
sys.path.insert(0, '/home/emilio/Documents/ai/NeuronMap/src/data_generation')

# Import the module
import domain_specialists

print("=== domain_specialists module contents ===")
print(f"Module file: {domain_specialists.__file__}")
print(f"Available attributes:")
for name in sorted(dir(domain_specialists)):
    if not name.startswith('_'):
        obj = getattr(domain_specialists, name)
        print(f"  {name}: {type(obj)}")

# Check if specific classes exist
classes_to_check = ['DomainSpecialist', 'PhysicsSpecialist', 'DomainSpecialistFactory', 'DomainType']
print(f"\n=== Checking specific classes ===")
for cls_name in classes_to_check:
    if hasattr(domain_specialists, cls_name):
        print(f"✅ {cls_name}: Found")
    else:
        print(f"❌ {cls_name}: Missing")
