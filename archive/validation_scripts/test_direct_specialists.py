#!/usr/bin/env python3
"""
Direct test of domain specialists from the main file
"""
import sys
import os
sys.path.insert(0, '/home/emilio/Documents/ai/NeuronMap/src/data_generation')

# Import directly from the domain_specialists.py file
import domain_specialists

def test_specialists():
    """Test direct creation and usage of specialists."""
    print("Testing direct specialist imports...")
    
    # Test Physics Specialist
    try:
        physics = domain_specialists.PhysicsSpecialist()
        print(f"✅ PhysicsSpecialist created: {physics.domain_type}")
        
        # Test question generation
        questions = physics.generate_questions(count=2)
        print(f"✅ Generated {len(questions)} physics questions")
        for i, q in enumerate(questions, 1):
            print(f"   {i}. {q['question'][:100]}...")
            
    except Exception as e:
        print(f"❌ PhysicsSpecialist error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Mathematics Specialist  
    try:
        math = domain_specialists.MathematicsSpecialist()
        print(f"✅ MathematicsSpecialist created: {math.domain_type}")
        
        questions = math.generate_questions(count=2)
        print(f"✅ Generated {len(questions)} math questions")
        for i, q in enumerate(questions, 1):
            print(f"   {i}. {q['question'][:100]}...")
            
    except Exception as e:
        print(f"❌ MathematicsSpecialist error: {e}")

    # Test Chemistry Specialist
    try:
        chem = domain_specialists.ChemistrySpecialist()
        print(f"✅ ChemistrySpecialist created: {chem.domain_type}")
        
        questions = chem.generate_questions(count=2)
        print(f"✅ Generated {len(questions)} chemistry questions")
        for i, q in enumerate(questions, 1):
            print(f"   {i}. {q['question'][:100]}...")
            
    except Exception as e:
        print(f"❌ ChemistrySpecialist error: {e}")

    # Test Factory
    try:
        available_domains = domain_specialists.DomainSpecialistFactory.get_available_domains()
        print(f"✅ Available domains: {len(available_domains)}")
        print(f"   Domains: {available_domains}")
        
        # Test creating specialist through factory
        physics_via_factory = domain_specialists.DomainSpecialistFactory.create_specialist(domain_specialists.DomainType.PHYSICS)
        print(f"✅ Created specialist via factory: {physics_via_factory.domain_type}")
        
    except Exception as e:
        print(f"❌ DomainSpecialistFactory error: {e}")

if __name__ == "__main__":
    test_specialists()
