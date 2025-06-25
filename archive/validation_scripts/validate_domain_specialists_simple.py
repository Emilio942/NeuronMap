#!/usr/bin/env python3
"""
Simple validation script for Section 4.1 Domain Specialists.
Validates the core functionality of the 20 domain specialists.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def validate_domain_specialists():
    """Validate the domain specialists implementation."""
    print("🔍 Validating Domain Specialists (Section 4.1)")
    print("=" * 60)
    
    try:
        from src.data_generation.domain_specialists import (
            get_available_domains,
            create_domain_specialist,
            generate_domain_specific_questions,
            DOMAIN_SPECIALISTS
        )
        print("✅ Successfully imported domain specialists")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 1: Check domain count
    print(f"\n1. Domain Coverage:")
    domains = get_available_domains()
    print(f"   Found {len(domains)} domains")
    if len(domains) >= 20:
        print("   ✅ Meets requirement (20+ domains)")
    else:
        print(f"   ❌ Below requirement (need 20, have {len(domains)})")
        return False
    
    # Test 2: List all domains
    print(f"\n2. Available Domains:")
    for i, domain in enumerate(sorted(domains), 1):
        print(f"   {i:2d}. {domain}")
    
    # Test 3: Test each specialist
    print(f"\n3. Specialist Functionality:")
    failed_domains = []
    
    for domain in sorted(domains):
        try:
            specialist = create_domain_specialist(domain)
            questions = specialist.generate_questions(count=2)
            vocabulary = specialist.get_domain_vocabulary()
            patterns = specialist.get_question_patterns()
            
            if len(questions) == 2 and len(vocabulary) > 0 and len(patterns) > 0:
                print(f"   ✅ {domain}: Generated {len(questions)} questions")
            else:
                print(f"   ❌ {domain}: Issues with generation")
                failed_domains.append(domain)
                
        except Exception as e:
            print(f"   ❌ {domain}: Error - {e}")
            failed_domains.append(domain)
    
    # Test 4: Bulk question generation
    print(f"\n4. Bulk Question Generation:")
    try:
        test_domains = domains[:5]  # Test first 5 domains
        questions_dict = generate_domain_specific_questions(
            domains=test_domains, 
            questions_per_domain=3
        )
        
        total_questions = sum(len(qs) for qs in questions_dict.values())
        print(f"   Generated {total_questions} questions for {len(test_domains)} domains")
        
        if total_questions >= len(test_domains) * 2:  # At least 2 questions per domain
            print("   ✅ Bulk generation working")
        else:
            print("   ❌ Bulk generation issues")
            
    except Exception as e:
        print(f"   ❌ Bulk generation failed: {e}")
    
    # Test 5: Sample questions
    print(f"\n5. Sample Questions:")
    sample_domains = ['science', 'mathematics', 'literature']
    for domain in sample_domains:
        if domain in domains:
            try:
                specialist = create_domain_specialist(domain)
                questions = specialist.generate_questions(count=2)
                print(f"   {domain.title()}:")
                for q in questions:
                    print(f"     - {q}")
            except Exception as e:
                print(f"   {domain}: Error - {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print(f"   Total Domains: {len(domains)}")
    print(f"   Failed Domains: {len(failed_domains)}")
    print(f"   Success Rate: {((len(domains) - len(failed_domains)) / len(domains) * 100):.1f}%")
    
    if len(failed_domains) == 0 and len(domains) >= 20:
        print("   🎉 ALL TESTS PASSED - Section 4.1 Core Requirements Met!")
        return True
    else:
        print("   ⚠️  Some issues remain")
        if failed_domains:
            print(f"   Failed domains: {', '.join(failed_domains)}")
        return False

if __name__ == "__main__":
    success = validate_domain_specialists()
    sys.exit(0 if success else 1)
