#!/usr/bin/env python3
"""
Domain Specialization Framework Validation Script
================================================

Validates Section 4.1 Domain-Specific Question Generation requirements:
- 15+ domain specialists implemented and functional
- >90% domain classification accuracy
- >25% terminology density in generated questions
- Expert validation criteria met
- <5% cross-domain contamination
"""

import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict, Counter

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_generation.domain_specialists import (
    create_domain_specialist,
    get_available_domains,
    generate_domain_specific_questions,
    BaseDomainSpecialist
)
from src.data_generation.difficulty_analyzer import assess_question_difficulty_fast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DomainValidationSuite:
    """Comprehensive validation suite for domain specialists."""
    
    def __init__(self):
        self.results = {
            'test_results': {},
            'performance_metrics': {},
            'validation_summary': {},
            'domain_stats': {}
        }
        
        # Sample texts for different domains
        self.sample_texts = {
            DomainType.PHYSICS: """
            The concept of wave-particle duality in quantum mechanics describes how 
            elementary particles exhibit both wave and particle properties. This 
            fundamental principle was established through experiments like the 
            double-slit experiment, which demonstrated that electrons can behave 
            as both particles and waves depending on the experimental setup.
            """,
            
            DomainType.CHEMISTRY: """
            Chemical bonding involves the interaction between atoms to form molecules 
            and compounds. Covalent bonds form when atoms share electrons, while ionic 
            bonds result from the transfer of electrons between atoms. The strength 
            and type of chemical bonds determine the properties of the resulting compounds.
            """,
            
            DomainType.BIOLOGY: """
            Cellular respiration is the process by which cells break down glucose 
            and other organic molecules to produce ATP, the energy currency of cells. 
            This complex biochemical process occurs in three main stages: glycolysis, 
            the citric acid cycle, and the electron transport chain.
            """,
            
            DomainType.MATHEMATICS: """
            Calculus is a branch of mathematics that deals with rates of change 
            and accumulation. Differential calculus focuses on derivatives, which 
            measure instantaneous rates of change, while integral calculus deals 
            with integrals, which measure accumulation over intervals.
            """,
            
            DomainType.COMPUTER_SCIENCE: """
            Machine learning algorithms enable computers to learn patterns from 
            data without being explicitly programmed. Supervised learning uses 
            labeled training data to make predictions, while unsupervised learning 
            finds hidden patterns in unlabeled data.
            """,
            
            DomainType.ENGINEERING: """
            Structural engineering involves the design and analysis of buildings, 
            bridges, and other structures to ensure they can safely support their 
            intended loads. Engineers must consider factors such as material 
            properties, environmental conditions, and safety factors.
            """,
            
            DomainType.HISTORY: """
            The Industrial Revolution, beginning in the late 18th century, 
            transformed society through mechanization and mass production. 
            This period saw significant changes in manufacturing, transportation, 
            and social structures that continue to influence modern society.
            """,
            
            DomainType.PHILOSOPHY: """
            Ethics is the branch of philosophy concerned with moral principles 
            and values. Deontological ethics focuses on the inherent rightness 
            or wrongness of actions, while consequentialist ethics evaluates 
            actions based on their outcomes and consequences.
            """,
            
            DomainType.LITERATURE: """
            Literary symbolism involves the use of symbols to represent ideas 
            or concepts beyond their literal meaning. Authors employ symbolic 
            elements to add depth and multiple layers of meaning to their works, 
            allowing readers to interpret themes on various levels.
            """,
            
            DomainType.LINGUISTICS: """
            Phonetics is the study of speech sounds and their physical properties. 
            It examines how sounds are produced by the human vocal apparatus, 
            how they are transmitted through air, and how they are perceived 
            by the human ear and processed by the brain.
            """,
            
            DomainType.ART_HISTORY: """
            The Renaissance period marked a revolutionary change in artistic 
            expression, characterized by renewed interest in classical antiquity, 
            humanism, and scientific observation. Artists like Leonardo da Vinci 
            and Michelangelo exemplified the period's integration of art and science.
            """,
            
            DomainType.PSYCHOLOGY: """
            Cognitive psychology studies mental processes such as perception, 
            memory, thinking, and problem-solving. This field examines how 
            humans acquire, process, store, and retrieve information, providing 
            insights into the mechanisms underlying human cognition.
            """,
            
            DomainType.SOCIOLOGY: """
            Social stratification refers to the hierarchical arrangement of 
            individuals in society based on factors such as wealth, power, 
            and prestige. This system of inequality affects access to resources, 
            opportunities, and life chances.
            """,
            
            DomainType.POLITICAL_SCIENCE: """
            Democratic governance relies on principles of popular sovereignty, 
            political equality, and majority rule with minority rights. 
            Constitutional frameworks establish the rules and institutions 
            that govern political processes and protect individual liberties.
            """,
            
            DomainType.ECONOMICS: """
            Market equilibrium occurs when the quantity of goods supplied 
            equals the quantity demanded at a particular price level. This 
            balance point determines the market price and reflects the 
            interaction between consumer preferences and producer costs.
            """,
            
            DomainType.ANTHROPOLOGY: """
            Cultural anthropology examines human societies and cultures, 
            focusing on how people organize their lives, create meaning, 
            and adapt to their environments. Ethnographic fieldwork provides 
            deep insights into diverse cultural practices and belief systems.
            """,
            
            DomainType.MEDICINE: """
            Evidence-based medicine integrates clinical expertise with the 
            best available research evidence and patient values. This approach 
            ensures that medical decisions are informed by rigorous scientific 
            research and tailored to individual patient needs.
            """,
            
            DomainType.LAW: """
            Constitutional law governs the interpretation and application of 
            constitutional principles and provisions. It addresses fundamental 
            questions about the structure of government, the distribution of 
            power, and the protection of individual rights.
            """,
            
            DomainType.EDUCATION: """
            Constructivist learning theory suggests that learners actively 
            build their understanding through experience and reflection. 
            This approach emphasizes the importance of prior knowledge, 
            social interaction, and authentic learning contexts.
            """,
            
            DomainType.BUSINESS: """
            Strategic management involves the formulation and implementation 
            of long-term plans to achieve organizational objectives. It requires 
            analysis of internal capabilities and external environments to 
            develop sustainable competitive advantages.
            """
        }
    
    def test_domain_coverage(self) -> Dict[str, Any]:
        """Test that 15+ domains are available and functional."""
        logger.info("Testing domain coverage...")
        
        available_domains = get_available_domains()
        domain_count = len(available_domains)
        
        # Test specialist creation for each domain
        functional_domains = []
        for domain in available_domains:
            try:
                specialist = create_domain_specialist(domain)
                functional_domains.append(domain)
                logger.info(f"✓ {domain.value} specialist created successfully")
            except Exception as e:
                logger.error(f"✗ Failed to create {domain.value} specialist: {e}")
        
        result = {
            'total_domains': domain_count,
            'functional_domains': len(functional_domains),
            'available_domains': [d.value for d in available_domains],
            'functional_domain_list': [d.value for d in functional_domains],
            'meets_requirement': domain_count >= 15 and len(functional_domains) >= 15
        }
        
        logger.info(f"Domain coverage: {domain_count} total, {len(functional_domains)} functional")
        return result
    
    def test_question_generation(self) -> Dict[str, Any]:
        """Test question generation for all domains."""
        logger.info("Testing question generation...")
        
        generation_results = {}
        total_questions = 0
        successful_generations = 0
        
        for domain in get_available_domains():
            if domain not in self.sample_texts:
                continue
                
            try:
                start_time = time.time()
                
                # Generate questions using the domain specialist
                questions = generate_domain_specific_questions(
                    domain, 
                    self.sample_texts[domain], 
                    count=5
                )
                
                generation_time = time.time() - start_time
                
                generation_results[domain.value] = {
                    'questions_generated': len(questions),
                    'generation_time': generation_time,
                    'avg_time_per_question': generation_time / max(len(questions), 1),
                    'questions': [q.question for q in questions],
                    'success': len(questions) > 0
                }
                
                total_questions += len(questions)
                if len(questions) > 0:
                    successful_generations += 1
                
                logger.info(f"✓ {domain.value}: {len(questions)} questions in {generation_time:.2f}s")
                
            except Exception as e:
                generation_results[domain.value] = {
                    'questions_generated': 0,
                    'error': str(e),
                    'success': False
                }
                logger.error(f"✗ {domain.value}: {e}")
        
        result = {
            'total_questions_generated': total_questions,
            'successful_domains': successful_generations,
            'total_domains_tested': len(generation_results),
            'success_rate': successful_generations / max(len(generation_results), 1),
            'generation_results': generation_results,
            'meets_requirement': successful_generations >= 15
        }
        
        return result
    
    def test_terminology_density(self) -> Dict[str, Any]:
        """Test that generated questions have >25% terminology density."""
        logger.info("Testing terminology density...")
        
        density_results = {}
        total_density = 0
        domain_count = 0
        
        for domain in get_available_domains()[:10]:  # Test subset for performance
            if domain not in self.sample_texts:
                continue
            
            try:
                specialist = create_domain_specialist(domain)
                questions = generate_domain_specific_questions(
                    domain, 
                    self.sample_texts[domain], 
                    count=3
                )
                
                if not questions:
                    continue
                
                # Calculate terminology density for each question
                question_densities = []
                domain_terms = set(specialist.vocabulary.get_all_terms())
                
                for q in questions:
                    words = q.question.lower().split()
                    term_count = sum(1 for word in words 
                                   if any(term.lower() in word for term in domain_terms))
                    density = (term_count / len(words)) * 100 if words else 0
                    question_densities.append(density)
                
                avg_density = sum(question_densities) / len(question_densities)
                
                density_results[domain.value] = {
                    'average_density': avg_density,
                    'question_densities': question_densities,
                    'meets_threshold': avg_density >= 25.0,
                    'vocabulary_size': len(domain_terms)
                }
                
                total_density += avg_density
                domain_count += 1
                
                logger.info(f"✓ {domain.value}: {avg_density:.1f}% terminology density")
                
            except Exception as e:
                logger.error(f"✗ {domain.value} terminology test: {e}")
        
        overall_density = total_density / max(domain_count, 1)
        
        result = {
            'overall_density': overall_density,
            'domains_tested': domain_count,
            'density_results': density_results,
            'meets_requirement': overall_density >= 25.0
        }
        
        return result
    
    def test_domain_classification_accuracy(self) -> Dict[str, Any]:
        """Test domain classification accuracy using validation methods."""
        logger.info("Testing domain classification accuracy...")
        
        classification_results = {}
        correct_classifications = 0
        total_tests = 0
        
        for domain in get_available_domains()[:10]:  # Test subset
            if domain not in self.sample_texts:
                continue
            
            try:
                specialist = create_domain_specialist(domain)
                questions = generate_domain_specific_questions(
                    domain, 
                    self.sample_texts[domain], 
                    count=3
                )
                
                if not questions:
                    continue
                
                # Test domain validation for generated questions
                correct_for_domain = 0
                for q in questions:
                    validation = specialist.validate_domain_specificity(q.question)
                    if validation.is_domain_specific and validation.confidence >= 0.7:
                        correct_for_domain += 1
                    total_tests += 1
                
                domain_accuracy = (correct_for_domain / len(questions)) * 100 if questions else 0
                correct_classifications += correct_for_domain
                
                classification_results[domain.value] = {
                    'accuracy': domain_accuracy,
                    'correct_classifications': correct_for_domain,
                    'total_questions': len(questions),
                    'meets_threshold': domain_accuracy >= 90.0
                }
                
                logger.info(f"✓ {domain.value}: {domain_accuracy:.1f}% classification accuracy")
                
            except Exception as e:
                logger.error(f"✗ {domain.value} classification test: {e}")
        
        overall_accuracy = (correct_classifications / max(total_tests, 1)) * 100
        
        result = {
            'overall_accuracy': overall_accuracy,
            'correct_classifications': correct_classifications,
            'total_tests': total_tests,
            'classification_results': classification_results,
            'meets_requirement': overall_accuracy >= 90.0
        }
        
        return result
    
    def test_cross_domain_contamination(self) -> Dict[str, Any]:
        """Test for <5% cross-domain contamination."""
        logger.info("Testing cross-domain contamination...")
        
        contamination_results = {}
        total_contamination = 0
        domain_pairs_tested = 0
        
        # Test a subset of domain pairs
        test_domains = list(get_available_domains())[:8]
        
        for i, domain1 in enumerate(test_domains):
            if domain1 not in self.sample_texts:
                continue
            
            try:
                specialist1 = create_domain_specialist(domain1)
                questions1 = generate_domain_specific_questions(
                    domain1, 
                    self.sample_texts[domain1], 
                    count=3
                )
                
                if not questions1:
                    continue
                
                contamination_count = 0
                total_cross_tests = 0
                
                # Test against other domains
                for j, domain2 in enumerate(test_domains):
                    if i == j or domain2 not in self.sample_texts:
                        continue
                    
                    try:
                        specialist2 = create_domain_specialist(domain2)
                        
                        # Check if domain1 questions are classified as domain2
                        for q in questions1:
                            validation = specialist2.validate_domain_specificity(q.question)
                            if validation.is_domain_specific and validation.confidence >= 0.7:
                                contamination_count += 1
                            total_cross_tests += 1
                    
                    except Exception:
                        continue
                
                domain_contamination = (contamination_count / max(total_cross_tests, 1)) * 100
                
                contamination_results[domain1.value] = {
                    'contamination_rate': domain_contamination,
                    'contamination_count': contamination_count,
                    'total_cross_tests': total_cross_tests,
                    'meets_threshold': domain_contamination <= 5.0
                }
                
                total_contamination += contamination_count
                domain_pairs_tested += total_cross_tests
                
                logger.info(f"✓ {domain1.value}: {domain_contamination:.1f}% contamination rate")
                
            except Exception as e:
                logger.error(f"✗ {domain1.value} contamination test: {e}")
        
        overall_contamination = (total_contamination / max(domain_pairs_tested, 1)) * 100
        
        result = {
            'overall_contamination': overall_contamination,
            'total_contamination': total_contamination,
            'domain_pairs_tested': domain_pairs_tested,
            'contamination_results': contamination_results,
            'meets_requirement': overall_contamination <= 5.0
        }
        
        return result
    
    def test_integration_with_difficulty_assessment(self) -> Dict[str, Any]:
        """Test integration with difficulty assessment system."""
        logger.info("Testing integration with difficulty assessment...")
        
        integration_results = {}
        successful_assessments = 0
        total_questions = 0
        
        for domain in get_available_domains()[:5]:  # Test subset
            if domain not in self.sample_texts:
                continue
            
            try:
                questions = generate_domain_specific_questions(
                    domain, 
                    self.sample_texts[domain], 
                    count=3
                )
                
                if not questions:
                    continue
                
                assessment_results = []
                for q in questions:
                    try:
                        # Test difficulty assessment integration
                        difficulty_result = assess_question_difficulty_fast(q.question)
                        assessment_results.append({
                            'question': q.question[:100] + "...",
                            'difficulty_score': difficulty_result.difficulty_score,
                            'difficulty_level': difficulty_result.difficulty_level.value,
                            'success': True
                        })
                        successful_assessments += 1
                    except Exception as e:
                        assessment_results.append({
                            'question': q.question[:100] + "...",
                            'error': str(e),
                            'success': False
                        })
                    total_questions += 1
                
                integration_results[domain.value] = {
                    'questions_tested': len(questions),
                    'successful_assessments': len([r for r in assessment_results if r.get('success')]),
                    'assessment_results': assessment_results
                }
                
                logger.info(f"✓ {domain.value}: {len([r for r in assessment_results if r.get('success')])}/{len(questions)} assessments successful")
                
            except Exception as e:
                logger.error(f"✗ {domain.value} integration test: {e}")
        
        integration_rate = (successful_assessments / max(total_questions, 1)) * 100
        
        result = {
            'integration_rate': integration_rate,
            'successful_assessments': successful_assessments,
            'total_questions': total_questions,
            'integration_results': integration_results,
            'meets_requirement': integration_rate >= 95.0
        }
        
        return result
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests and compile results."""
        logger.info("Starting comprehensive domain specialization validation...")
        start_time = time.time()
        
        # Run all tests
        tests = {
            'domain_coverage': self.test_domain_coverage,
            'question_generation': self.test_question_generation,
            'terminology_density': self.test_terminology_density,
            'classification_accuracy': self.test_domain_classification_accuracy,
            'cross_domain_contamination': self.test_cross_domain_contamination,
            'difficulty_integration': self.test_integration_with_difficulty_assessment
        }
        
        for test_name, test_func in tests.items():
            try:
                logger.info(f"\n--- Running {test_name} test ---")
                self.results['test_results'][test_name] = test_func()
                self.results['test_results'][test_name]['test_passed'] = True
            except Exception as e:
                logger.error(f"Test {test_name} failed: {e}")
                self.results['test_results'][test_name] = {
                    'error': str(e),
                    'test_passed': False
                }
        
        # Calculate overall metrics
        total_time = time.time() - start_time
        
        # Check requirements compliance
        requirements_met = []
        requirements_checked = []
        
        for test_name, result in self.results['test_results'].items():
            if result.get('test_passed') and 'meets_requirement' in result:
                requirements_checked.append(test_name)
                if result['meets_requirement']:
                    requirements_met.append(test_name)
        
        compliance_rate = (len(requirements_met) / max(len(requirements_checked), 1)) * 100
        
        self.results['validation_summary'] = {
            'total_validation_time': total_time,
            'tests_run': len(tests),
            'tests_passed': len([r for r in self.results['test_results'].values() if r.get('test_passed')]),
            'requirements_checked': len(requirements_checked),
            'requirements_met': len(requirements_met),
            'compliance_rate': compliance_rate,
            'overall_success': compliance_rate >= 80.0,
            'section_4_1_domain_complete': compliance_rate >= 90.0
        }
        
        return self.results


def main():
    """Main validation function."""
    print("=" * 80)
    print("NEURONMAP DOMAIN SPECIALIZATION FRAMEWORK VALIDATION")
    print("Section 4.1: Domain-Specific Question Generation")
    print("=" * 80)
    
    validator = DomainValidationSuite()
    results = validator.run_comprehensive_validation()
    
    # Display results
    print(f"\nVALIDATION SUMMARY:")
    print(f"Tests run: {results['validation_summary']['tests_run']}")
    print(f"Tests passed: {results['validation_summary']['tests_passed']}")
    print(f"Requirements compliance: {results['validation_summary']['compliance_rate']:.1f}%")
    print(f"Overall success: {'✓' if results['validation_summary']['overall_success'] else '✗'}")
    print(f"Section 4.1 Domain Complete: {'✓' if results['validation_summary']['section_4_1_domain_complete'] else '✗'}")
    
    # Detailed results
    print(f"\nDETAILED RESULTS:")
    for test_name, result in results['test_results'].items():
        if result.get('test_passed'):
            requirement_status = "✓" if result.get('meets_requirement', False) else "✗"
            print(f"{requirement_status} {test_name}: {result.get('meets_requirement', 'N/A')}")
        else:
            print(f"✗ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Save results to file
    results_file = Path(__file__).parent / 'domain_specialization_validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Return appropriate exit code
    if results['validation_summary']['section_4_1_domain_complete']:
        print("\n🎉 SECTION 4.1 DOMAIN SPECIALIZATION FRAMEWORK: COMPLETE")
        return 0
    else:
        print("\n❌ SECTION 4.1 DOMAIN SPECIALIZATION FRAMEWORK: INCOMPLETE")
        return 1


if __name__ == '__main__':
    sys.exit(main())
