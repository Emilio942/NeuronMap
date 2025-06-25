#!/usr/bin/env python3
"""
Domain Specialization Framework Validation Script
================================================

Comprehensive validation of Section 4.1 Domain-Specific Question Generation.
Tests all domain specialists with authentic domain-specific requirements.
"""

import sys
import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.data_generation.domain_specialization_framework import (
        get_available_domains, create_domain_specialist, DomainType, QuestionType
    )
    from src.data_generation.difficulty_analyzer import assess_question_difficulty_fast
except ImportError as e:
    print(f"Import error: {e}")
    print("Running validation without all modules...")


class DomainSpecializationValidator:
    """Comprehensive validator for domain specialization framework."""
    
    def __init__(self):
        self.results = {
            'total_domains': 0,
            'functional_domains': 0,
            'questions_generated': 0,
            'validation_passed': 0,
            'terminology_density_avg': 0.0,
            'domain_accuracy': {},
            'performance_metrics': {},
            'detailed_results': {}
        }
        
        self.test_topics = {
            'physics': ['quantum mechanics', 'relativity', 'thermodynamics', 'electromagnetism'],
            'chemistry': ['organic chemistry', 'inorganic chemistry', 'physical chemistry', 'analytical chemistry'],
            'biology': ['molecular biology', 'ecology', 'genetics', 'evolution'],
            'mathematics': ['calculus', 'algebra', 'geometry', 'statistics'],
            'computer_science': ['algorithms', 'machine learning', 'databases', 'software engineering'],
            'engineering': ['mechanical engineering', 'electrical engineering', 'civil engineering'],
            'history': ['ancient history', 'medieval history', 'modern history', 'contemporary history'],
            'philosophy': ['ethics', 'metaphysics', 'epistemology', 'logic'],
            'literature': ['poetry', 'prose', 'drama', 'literary criticism'],
            'linguistics': ['phonetics', 'syntax', 'semantics', 'pragmatics'],
            'art_history': ['renaissance art', 'modern art', 'architecture', 'sculpture'],
            'psychology': ['cognitive psychology', 'developmental psychology', 'social psychology'],
            'sociology': ['social theory', 'social institutions', 'social inequality'],
            'political_science': ['political theory', 'comparative politics', 'international relations'],
            'economics': ['microeconomics', 'macroeconomics', 'international economics'],
            'anthropology': ['cultural anthropology', 'archaeology', 'linguistic anthropology'],
            'medicine': ['cardiology', 'neurology', 'oncology', 'infectious disease'],
            'law': ['constitutional law', 'criminal law', 'civil law', 'international law'],
            'education': ['curriculum theory', 'educational psychology', 'educational technology'],
            'business': ['strategic management', 'marketing', 'finance', 'operations']
        }
    
    def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of domain specialization framework."""
        print("=" * 80)
        print("DOMAIN SPECIALIZATION FRAMEWORK VALIDATION")
        print("=" * 80)
        
        # Test 1: Domain Coverage
        print("\n1. Testing Domain Coverage...")
        self._test_domain_coverage()
        
        # Test 2: Specialist Functionality
        print("\n2. Testing Specialist Functionality...")
        self._test_specialist_functionality()
        
        # Test 3: Question Quality
        print("\n3. Testing Question Quality...")
        self._test_question_quality()
        
        # Test 4: Domain Specificity
        print("\n4. Testing Domain Specificity...")
        self._test_domain_specificity()
        
        # Test 5: Performance Metrics
        print("\n5. Testing Performance Metrics...")
        self._test_performance()
        
        # Test 6: Integration Test
        print("\n6. Testing Integration...")
        self._test_integration()
        
        # Generate final report
        self._generate_report()
        
        return self.results
    
    def _test_domain_coverage(self):
        """Test that we have 15+ domain specialists."""
        try:
            available_domains = get_available_domains()
            self.results['total_domains'] = len(available_domains)
            
            print(f"Available domains: {len(available_domains)}")
            print(f"Domains: {', '.join(available_domains)}")
            
            # Requirement: 15+ domains
            if len(available_domains) >= 15:
                print("✅ Domain coverage requirement met (15+ domains)")
            else:
                print(f"❌ Domain coverage insufficient: {len(available_domains)}/15 required")
                
        except Exception as e:
            print(f"❌ Error testing domain coverage: {e}")
            self.results['total_domains'] = 0
    
    def _test_specialist_functionality(self):
        """Test that domain specialists can be created and are functional."""
        functional_count = 0
        
        try:
            available_domains = get_available_domains()
            
            for domain in available_domains:
                try:
                    specialist = create_domain_specialist(domain)
                    if specialist is not None:
                        # Test basic functionality
                        vocab = specialist.get_domain_vocabulary()
                        stats = specialist.get_generation_statistics()
                        
                        # Verify vocabulary has content
                        if (len(vocab.core_terms) > 0 and 
                            len(vocab.advanced_terms) > 0 and
                            hasattr(specialist, 'question_patterns')):
                            functional_count += 1
                            print(f"✅ {domain}: Functional specialist")
                        else:
                            print(f"❌ {domain}: Incomplete specialist implementation")
                    else:
                        print(f"❌ {domain}: Could not create specialist")
                        
                except Exception as e:
                    print(f"❌ {domain}: Error creating specialist - {e}")
            
            self.results['functional_domains'] = functional_count
            print(f"\nFunctional specialists: {functional_count}/{len(available_domains)}")
            
        except Exception as e:
            print(f"❌ Error testing specialist functionality: {e}")
    
    def _test_question_quality(self):
        """Test quality of generated questions."""
        try:
            available_domains = get_available_domains()
            total_questions = 0
            quality_scores = []
            
            for domain in available_domains[:10]:  # Test first 10 domains
                try:
                    specialist = create_domain_specialist(domain)
                    if specialist is None:
                        continue
                    
                    # Test with domain-specific topics
                    topics = self.test_topics.get(domain, [domain])[:2]  # First 2 topics
                    
                    for topic in topics:
                        questions = specialist.generate_domain_questions(topic, count=3)
                        total_questions += len(questions)
                        
                        for question in questions:
                            # Assess quality metrics
                            quality_score = self._assess_question_quality(question)
                            quality_scores.append(quality_score)
                
                except Exception as e:
                    print(f"❌ Error testing {domain}: {e}")
            
            self.results['questions_generated'] = total_questions
            if quality_scores:
                avg_quality = statistics.mean(quality_scores)
                self.results['average_quality'] = avg_quality
                print(f"Generated {total_questions} questions")
                print(f"Average quality score: {avg_quality:.2f}/10")
                
                if avg_quality >= 7.0:
                    print("✅ Question quality meets standards")
                else:
                    print("❌ Question quality below standards")
            
        except Exception as e:
            print(f"❌ Error testing question quality: {e}")
    
    def _test_domain_specificity(self):
        """Test domain specificity of generated questions."""
        try:
            terminology_densities = []
            domain_validations = []
            
            available_domains = get_available_domains()
            
            for domain in available_domains[:8]:  # Test subset for performance
                try:
                    specialist = create_domain_specialist(domain)
                    if specialist is None:
                        continue
                    
                    topics = self.test_topics.get(domain, [domain])[:1]
                    
                    for topic in topics:
                        questions = specialist.generate_domain_questions(topic, count=5)
                        
                        for question in questions:
                            # Test terminology density
                            terminology_density = question.domain_complexity.terminology_density
                            terminology_densities.append(terminology_density)
                            
                            # Test domain validation
                            validation_passed = question.validation_result.is_domain_appropriate
                            domain_validations.append(validation_passed)
                
                except Exception as e:
                    print(f"❌ Error testing {domain} specificity: {e}")
            
            if terminology_densities:
                avg_density = statistics.mean(terminology_densities)
                self.results['terminology_density_avg'] = avg_density
                print(f"Average terminology density: {avg_density:.3f}")
                
                # Requirement: >25% terminology density
                if avg_density >= 0.25:
                    print("✅ Terminology density meets requirement (>25%)")
                else:
                    print(f"❌ Terminology density below requirement: {avg_density:.1%}/25%")
            
            if domain_validations:
                validation_rate = sum(domain_validations) / len(domain_validations)
                self.results['validation_pass_rate'] = validation_rate
                print(f"Domain validation pass rate: {validation_rate:.1%}")
                
                # Requirement: >90% domain classification accuracy
                if validation_rate >= 0.90:
                    print("✅ Domain validation meets requirement (>90%)")
                else:
                    print(f"❌ Domain validation below requirement: {validation_rate:.1%}/90%")
        
        except Exception as e:
            print(f"❌ Error testing domain specificity: {e}")
    
    def _test_performance(self):
        """Test performance metrics."""
        try:
            available_domains = get_available_domains()
            generation_times = []
            
            for domain in available_domains[:5]:  # Test subset
                try:
                    specialist = create_domain_specialist(domain)
                    if specialist is None:
                        continue
                    
                    # Measure generation time
                    start_time = time.time()
                    questions = specialist.generate_domain_questions("test topic", count=10)
                    generation_time = time.time() - start_time
                    
                    avg_time_per_question = generation_time / len(questions) if questions else float('inf')
                    generation_times.append(avg_time_per_question)
                    
                except Exception as e:
                    print(f"❌ Error testing {domain} performance: {e}")
            
            if generation_times:
                avg_generation_time = statistics.mean(generation_times) * 1000  # Convert to ms
                self.results['avg_generation_time_ms'] = avg_generation_time
                
                print(f"Average generation time: {avg_generation_time:.2f}ms per question")
                
                # Reasonable performance expectation
                if avg_generation_time < 500:  # 500ms per question
                    print("✅ Generation performance acceptable")
                else:
                    print("❌ Generation performance slow")
        
        except Exception as e:
            print(f"❌ Error testing performance: {e}")
    
    def _test_integration(self):
        """Test integration with difficulty assessment."""
        try:
            # Test a few specialists with difficulty integration
            test_domains = ['physics', 'computer_science', 'history', 'medicine']
            integrated_successfully = 0
            
            for domain in test_domains:
                try:
                    specialist = create_domain_specialist(domain)
                    if specialist is None:
                        continue
                    
                    questions = specialist.generate_domain_questions("test integration", count=3)
                    
                    # Check that difficulty assessment is integrated
                    for question in questions:
                        if (hasattr(question, 'difficulty_assessment') and 
                            question.difficulty_assessment is not None):
                            integrated_successfully += 1
                            break
                
                except Exception as e:
                    print(f"❌ Error testing {domain} integration: {e}")
            
            integration_rate = integrated_successfully / len(test_domains)
            self.results['integration_success_rate'] = integration_rate
            
            print(f"Integration success rate: {integration_rate:.1%}")
            
            if integration_rate >= 0.75:
                print("✅ Integration with difficulty assessment successful")
            else:
                print("❌ Integration issues detected")
        
        except Exception as e:
            print(f"❌ Error testing integration: {e}")
    
    def _assess_question_quality(self, question) -> float:
        """Assess the quality of a generated question."""
        quality_score = 0.0
        
        # Length check (reasonable question length)
        if 20 <= len(question.question) <= 200:
            quality_score += 2.0
        
        # Has domain terminology
        if len(question.terminology_used) > 0:
            quality_score += 2.0
        
        # Domain specificity
        if question.validation_result.is_domain_appropriate:
            quality_score += 2.0
        
        # Complexity assessment exists
        if question.domain_complexity.overall_score > 0:
            quality_score += 2.0
        
        # Grammar check (basic)
        if question.question.strip().endswith('?'):
            quality_score += 1.0
        
        # Has required concepts
        if len(question.concepts_required) > 0:
            quality_score += 1.0
        
        return quality_score
    
    def _generate_report(self):
        """Generate final validation report."""
        print("\n" + "=" * 80)
        print("FINAL VALIDATION REPORT")
        print("=" * 80)
        
        # Summary metrics
        print(f"Total domains implemented: {self.results.get('total_domains', 0)}")
        print(f"Functional domains: {self.results.get('functional_domains', 0)}")
        print(f"Questions generated: {self.results.get('questions_generated', 0)}")
        print(f"Average terminology density: {self.results.get('terminology_density_avg', 0):.3f}")
        print(f"Validation pass rate: {self.results.get('validation_pass_rate', 0):.1%}")
        print(f"Average generation time: {self.results.get('avg_generation_time_ms', 0):.2f}ms")
        print(f"Integration success rate: {self.results.get('integration_success_rate', 0):.1%}")
        
        # Pass/Fail Assessment
        print("\nREQUIREMENT ASSESSMENT:")
        
        requirements_met = 0
        total_requirements = 7
        
        # Requirement 1: 15+ domains
        if self.results.get('total_domains', 0) >= 15:
            print("✅ 1. Domain coverage (15+ domains)")
            requirements_met += 1
        else:
            print("❌ 1. Domain coverage insufficient")
        
        # Requirement 2: Functional specialists
        if self.results.get('functional_domains', 0) >= 15:
            print("✅ 2. Functional domain specialists")
            requirements_met += 1
        else:
            print("❌ 2. Some specialists non-functional")
        
        # Requirement 3: Question generation
        if self.results.get('questions_generated', 0) > 0:
            print("✅ 3. Question generation working")
            requirements_met += 1
        else:
            print("❌ 3. Question generation failed")
        
        # Requirement 4: Terminology density >25%
        if self.results.get('terminology_density_avg', 0) >= 0.25:
            print("✅ 4. Terminology density >25%")
            requirements_met += 1
        else:
            print("❌ 4. Terminology density below 25%")
        
        # Requirement 5: Domain validation >90%
        if self.results.get('validation_pass_rate', 0) >= 0.90:
            print("✅ 5. Domain validation >90%")
            requirements_met += 1
        else:
            print("❌ 5. Domain validation below 90%")
        
        # Requirement 6: Performance
        if self.results.get('avg_generation_time_ms', float('inf')) < 500:
            print("✅ 6. Acceptable performance")
            requirements_met += 1
        else:
            print("❌ 6. Performance issues")
        
        # Requirement 7: Integration
        if self.results.get('integration_success_rate', 0) >= 0.75:
            print("✅ 7. Integration successful")
            requirements_met += 1
        else:
            print("❌ 7. Integration issues")
        
        # Overall assessment
        print(f"\nOVERALL: {requirements_met}/{total_requirements} requirements met")
        
        if requirements_met >= 6:
            print("🎉 SECTION 4.1 DOMAIN-SPECIFIC QUESTION GENERATION: PASSED")
            self.results['overall_status'] = 'PASSED'
        else:
            print("❌ SECTION 4.1 DOMAIN-SPECIFIC QUESTION GENERATION: NEEDS WORK")
            self.results['overall_status'] = 'NEEDS_WORK'
        
        # Save results
        with open('domain_specialization_validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed results saved to: domain_specialization_validation_results.json")


def main():
    """Run the domain specialization validation."""
    validator = DomainSpecializationValidator()
    results = validator.run_validation()
    
    return results['overall_status'] == 'PASSED'


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
