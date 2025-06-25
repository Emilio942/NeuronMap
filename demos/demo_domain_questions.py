#!/usr/bin/env python3
"""Demo script for domain-specific question generation."""

import json
import logging
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generation.domain_specific_generator import (
    DomainSpecificQuestionGenerator, Domain, DifficultyLevel, QuestionCategory
)


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def demo_single_domain_generation():
    """Demonstrate single domain question generation."""
    print("\n" + "="*60)
    print("🔬 DEMO: Single Domain Question Generation")
    print("="*60)
    
    try:
        generator = DomainSpecificQuestionGenerator("default")
        
        # Test different domains
        test_cases = [
            (Domain.SCIENCE, DifficultyLevel.INTERMEDIATE, QuestionCategory.REASONING, 3),
            (Domain.MATHEMATICS, DifficultyLevel.ADVANCED, QuestionCategory.ANALYTICAL, 3),
            (Domain.HISTORY, DifficultyLevel.BEGINNER, QuestionCategory.FACTUAL, 2),
            (Domain.TECHNOLOGY, DifficultyLevel.EXPERT, QuestionCategory.TECHNICAL, 2)
        ]
        
        all_questions = []
        
        for domain, difficulty, category, num_questions in test_cases:
            print(f"\n🔸 Generating {num_questions} {difficulty.value} {category.value} questions for {domain.value}...")
            
            questions = generator.generate_domain_questions(
                domain, difficulty, category, num_questions
            )
            
            print(f"✅ Generated {len(questions)} questions")
            
            for i, q in enumerate(questions, 1):
                print(f"  {i}. [{q.difficulty.value.upper()}|{q.category.value.upper()}|{q.domain.upper()}]")
                print(f"     {q.text}")
                if q.keywords:
                    print(f"     Keywords: {', '.join(q.keywords[:4])}")
                print()
            
            all_questions.extend(questions)
        
        # Save single domain questions
        output_file = "data/demo_single_domain_questions.json"
        success = generator.save_questions_with_metadata(all_questions, output_file)
        
        if success:
            print(f"💾 Saved {len(all_questions)} single domain questions to {output_file}")
        
        return all_questions
        
    except Exception as e:
        print(f"❌ Single domain generation demo failed: {e}")
        return []


def demo_multi_domain_generation():
    """Demonstrate multi-domain question generation."""
    print("\n" + "="*60)
    print("🌍 DEMO: Multi-Domain Question Generation")
    print("="*60)
    
    try:
        generator = DomainSpecificQuestionGenerator("default")
        
        # Test multi-domain generation
        domains = [Domain.SCIENCE, Domain.TECHNOLOGY, Domain.MATHEMATICS, Domain.PHILOSOPHY]
        total_questions = 20
        
        print(f"🔸 Generating {total_questions} questions across {len(domains)} domains...")
        print(f"   Domains: {[d.value for d in domains]}")
        
        questions = generator.generate_multi_domain_questions(domains, total_questions)
        
        if questions:
            print(f"✅ Generated {len(questions)} multi-domain questions")
            
            # Analyze the distribution
            domain_counts = {}
            difficulty_counts = {}
            category_counts = {}
            
            for q in questions:
                domain = q.domain or "unknown"
                diff = q.difficulty.value
                cat = q.category.value
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            print("\n📊 Distribution Analysis:")
            print(f"Domain distribution: {domain_counts}")
            print(f"Difficulty distribution: {difficulty_counts}")
            print(f"Category distribution: {category_counts}")
            
            # Show sample questions from each domain
            print("\n📝 Sample Questions by Domain:")
            shown_domains = set()
            for q in questions:
                if q.domain and q.domain not in shown_domains:
                    print(f"  {q.domain.upper()}: [{q.difficulty.value.upper()}] {q.text}")
                    if q.keywords:
                        print(f"    Keywords: {', '.join(q.keywords[:3])}")
                    shown_domains.add(q.domain)
            
            # Save multi-domain questions
            output_file = "data/demo_multi_domain_questions.json"
            success = generator.save_questions_with_metadata(questions, output_file)
            
            if success:
                print(f"💾 Saved {len(questions)} multi-domain questions to {output_file}")
            
            return questions
        else:
            print("❌ No questions generated")
            return []
            
    except Exception as e:
        print(f"❌ Multi-domain generation demo failed: {e}")
        return []


def demo_subdomain_targeting():
    """Demonstrate subdomain targeting within domains."""
    print("\n" + "="*60)
    print("🎯 DEMO: Subdomain Targeting")
    print("="*60)
    
    try:
        generator = DomainSpecificQuestionGenerator("default")
        
        # Test subdomain targeting
        test_cases = [
            (Domain.SCIENCE, "biology", DifficultyLevel.INTERMEDIATE, QuestionCategory.REASONING, 3),
            (Domain.TECHNOLOGY, "artificial intelligence", DifficultyLevel.ADVANCED, QuestionCategory.TECHNICAL, 3),
            (Domain.MATHEMATICS, "calculus", DifficultyLevel.EXPERT, QuestionCategory.ANALYTICAL, 2)
        ]
        
        all_questions = []
        
        for domain, subdomain, difficulty, category, num_questions in test_cases:
            print(f"\n🔸 Generating {num_questions} {difficulty.value} {category.value} questions")
            print(f"   Domain: {domain.value}, Subdomain: {subdomain}")
            
            questions = generator.generate_domain_questions(
                domain, difficulty, category, num_questions, subdomain
            )
            
            print(f"✅ Generated {len(questions)} questions")
            
            for i, q in enumerate(questions, 1):
                print(f"  {i}. [{q.subdomain or 'GENERAL'}] {q.text}")
                # Check if subdomain keywords appear in the question
                subdomain_keywords = [word for word in subdomain.split() if word.lower() in q.text.lower()]
                if subdomain_keywords:
                    print(f"     Subdomain match: {', '.join(subdomain_keywords)}")
                if q.keywords:
                    print(f"     Keywords: {', '.join(q.keywords[:3])}")
                print()
            
            all_questions.extend(questions)
        
        # Save subdomain questions
        output_file = "data/demo_subdomain_questions.json"
        success = generator.save_questions_with_metadata(all_questions, output_file)
        
        if success:
            print(f"💾 Saved {len(all_questions)} subdomain questions to {output_file}")
        
        return all_questions
        
    except Exception as e:
        print(f"❌ Subdomain targeting demo failed: {e}")
        return []


def demo_domain_template_analysis():
    """Demonstrate analysis of domain templates and their effectiveness."""
    print("\n" + "="*60)
    print("📈 DEMO: Domain Template Analysis")
    print("="*60)
    
    try:
        generator = DomainSpecificQuestionGenerator("default")
        
        # Get all available domains
        available_domains = generator.prompt_manager.get_all_domains()
        
        print(f"📊 Available Domains ({len(available_domains)}):")
        for domain in available_domains:
            template = generator.prompt_manager.get_template(domain)
            print(f"\n  {domain.value.upper()}:")
            print(f"    Specialized concepts: {len(template.specialized_concepts)}")
            print(f"    Context keywords: {len(template.context_keywords)}")
            print(f"    Subdomain areas: {len(template.subdomain_areas)}")
            print(f"    Example questions per difficulty: {[len(examples) for examples in template.example_questions.values()]}")
            
            # Show some specialized concepts
            concepts = template.specialized_concepts[:5]
            print(f"    Sample concepts: {', '.join(concepts)}")
        
        # Test template effectiveness by generating questions from each domain
        print(f"\n🧪 Testing Template Effectiveness...")
        template_effectiveness = {}
        
        for domain in available_domains[:6]:  # Test first 6 domains
            try:
                questions = generator.generate_domain_questions(
                    domain, DifficultyLevel.INTERMEDIATE, QuestionCategory.REASONING, 2
                )
                
                if questions:
                    # Analyze domain relevance
                    template = generator.prompt_manager.get_template(domain)
                    domain_keywords = template.context_keywords + template.specialized_concepts
                    
                    relevance_scores = []
                    for q in questions:
                        question_lower = q.text.lower()
                        matches = sum(1 for keyword in domain_keywords if keyword.lower() in question_lower)
                        relevance_score = matches / len(domain_keywords) if domain_keywords else 0
                        relevance_scores.append(relevance_score)
                    
                    avg_relevance = sum(relevance_scores) / len(relevance_scores)
                    avg_quality = sum(q.quality_score for q in questions) / len(questions)
                    
                    template_effectiveness[domain.value] = {
                        'questions_generated': len(questions),
                        'avg_relevance': avg_relevance,
                        'avg_quality': avg_quality
                    }
                    
                    print(f"  ✅ {domain.value}: {len(questions)} questions, relevance: {avg_relevance:.3f}, quality: {avg_quality:.3f}")
                else:
                    template_effectiveness[domain.value] = {
                        'questions_generated': 0,
                        'avg_relevance': 0,
                        'avg_quality': 0
                    }
                    print(f"  ❌ {domain.value}: No questions generated")
                    
            except Exception as e:
                print(f"  ⚠️ {domain.value}: Error - {e}")
        
        # Print effectiveness summary
        print(f"\n📋 Template Effectiveness Summary:")
        for domain, metrics in template_effectiveness.items():
            if metrics['questions_generated'] > 0:
                print(f"  {domain}: Generated {metrics['questions_generated']}, "
                      f"Relevance {metrics['avg_relevance']:.2f}, Quality {metrics['avg_quality']:.2f}")
        
        return template_effectiveness
        
    except Exception as e:
        print(f"❌ Domain template analysis failed: {e}")
        return {}


def main():
    """Run all domain-specific demos."""
    print("🚀 NeuronMap Domain-Specific Question Generation Demo")
    print("=" * 60)
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    setup_logging()
    
    try:
        # Run demos
        single_domain_questions = demo_single_domain_generation()
        
        multi_domain_questions = demo_multi_domain_generation()
        
        subdomain_questions = demo_subdomain_targeting()
        
        template_analysis = demo_domain_template_analysis()
        
        print("\n" + "="*60)
        print("🎉 Domain-Specific Demo completed successfully!")
        print("="*60)
        print("\n📁 Generated files:")
        
        for file_path in [
            "data/demo_single_domain_questions.json", 
            "data/demo_multi_domain_questions.json",
            "data/demo_subdomain_questions.json"
        ]:
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                print(f"  {file_path} ({file_size:,} bytes)")
        
        print("\n💡 Next steps:")
        print("  1. Examine the generated JSON files to see domain-specific metadata")
        print("  2. Use domain-specific questions for targeted activation extraction")
        print("  3. Analyze how different knowledge domains affect neural representations")
        print("  4. Compare activation patterns across domains")
        print("  5. Experiment with subdomain targeting for fine-grained analysis")
        
        # Summary statistics
        total_questions = len(single_domain_questions) + len(multi_domain_questions) + len(subdomain_questions)
        print(f"\n📊 Total Questions Generated: {total_questions}")
        
        if template_analysis:
            successful_domains = sum(1 for metrics in template_analysis.values() if metrics['questions_generated'] > 0)
            print(f"📈 Successfully tested {successful_domains}/{len(template_analysis)} domain templates")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()