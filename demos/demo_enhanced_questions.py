#!/usr/bin/env python3
"""Demo script for enhanced question generation with difficulty control."""

import json
import logging
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generation.enhanced_question_generator import (
    EnhancedQuestionGenerator, DifficultyLevel, QuestionCategory
)


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def demo_targeted_generation():
    """Demonstrate targeted question generation for specific difficulty and category."""
    print("\n" + "="*60)
    print("🎯 DEMO: Targeted Question Generation")
    print("="*60)
    
    try:
        generator = EnhancedQuestionGenerator("default")
        
        # Generate questions for different combinations
        test_cases = [
            (DifficultyLevel.BEGINNER, QuestionCategory.FACTUAL, 3),
            (DifficultyLevel.EXPERT, QuestionCategory.REASONING, 3),
            (DifficultyLevel.INTERMEDIATE, QuestionCategory.CREATIVE, 2),
            (DifficultyLevel.ADVANCED, QuestionCategory.TECHNICAL, 2)
        ]
        
        all_questions = []
        
        for difficulty, category, num_questions in test_cases:
            print(f"\n🔸 Generating {num_questions} {difficulty.value} {category.value} questions...")
            
            questions = generator.generate_questions_with_difficulty(
                difficulty, category, num_questions
            )
            
            print(f"✅ Generated {len(questions)} questions")
            
            for i, q in enumerate(questions, 1):
                print(f"  {i}. [{q.difficulty.value.upper()}] {q.text}")
                print(f"     Category: {q.category.value}, Quality: {q.quality_score:.2f}, "
                      f"Complexity: {q.estimated_complexity_score:.2f}")
                if q.keywords:
                    print(f"     Keywords: {', '.join(q.keywords[:3])}")
                print()
            
            all_questions.extend(questions)
        
        # Save targeted questions
        output_file = "data/demo_targeted_questions.json"
        success = generator.save_questions_with_metadata(all_questions, output_file)
        
        if success:
            print(f"💾 Saved {len(all_questions)} targeted questions to {output_file}")
        
        return all_questions
        
    except Exception as e:
        print(f"❌ Targeted generation demo failed: {e}")
        return []


def demo_balanced_generation():
    """Demonstrate balanced question set generation."""
    print("\n" + "="*60)
    print("⚖️  DEMO: Balanced Question Set Generation")
    print("="*60)
    
    try:
        generator = EnhancedQuestionGenerator("default")
        
        print("🔸 Generating balanced question set (20 questions)...")
        
        questions = generator.generate_balanced_question_set(20)
        
        if questions:
            print(f"✅ Generated {len(questions)} balanced questions")
            
            # Analyze the distribution
            difficulty_counts = {}
            category_counts = {}
            
            for q in questions:
                diff = q.difficulty.value
                cat = q.category.value
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            print("\n📊 Distribution Analysis:")
            print(f"Difficulty distribution: {difficulty_counts}")
            print(f"Category distribution: {category_counts}")
            
            # Show some example questions
            print("\n📝 Sample Questions:")
            for i, q in enumerate(questions[:5], 1):
                print(f"  {i}. [{q.difficulty.value.upper()}|{q.category.value.upper()}] {q.text}")
                if q.domain:
                    print(f"     Domain: {q.domain}")
                print()
            
            # Save balanced questions
            output_file = "data/demo_balanced_questions.json"
            success = generator.save_questions_with_metadata(questions, output_file)
            
            if success:
                print(f"💾 Saved {len(questions)} balanced questions to {output_file}")
            
            return questions
        else:
            print("❌ No questions generated")
            return []
            
    except Exception as e:
        print(f"❌ Balanced generation demo failed: {e}")
        return []


def demo_metadata_analysis():
    """Demonstrate metadata analysis of generated questions."""
    print("\n" + "="*60)
    print("📊 DEMO: Metadata Analysis")
    print("="*60)
    
    # Load questions from previous demos
    question_files = [
        "data/demo_targeted_questions.json",
        "data/demo_balanced_questions.json"
    ]
    
    all_questions = []
    
    for file_path in question_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    questions = data.get('questions', [])
                    all_questions.extend(questions)
                    print(f"📁 Loaded {len(questions)} questions from {file_path}")
            except Exception as e:
                print(f"⚠️ Could not load {file_path}: {e}")
    
    if not all_questions:
        print("❌ No questions available for analysis")
        return
    
    print(f"\n🔍 Analyzing {len(all_questions)} questions...")
    
    # Analyze various metadata aspects
    print("\n📈 Quality Metrics:")
    quality_scores = [q['quality_score'] for q in all_questions]
    if quality_scores:
        print(f"  Average quality score: {sum(quality_scores)/len(quality_scores):.3f}")
        print(f"  Quality range: {min(quality_scores):.3f} - {max(quality_scores):.3f}")
    
    print("\n📏 Length Metrics:")
    word_counts = [q['word_count'] for q in all_questions]
    if word_counts:
        print(f"  Average word count: {sum(word_counts)/len(word_counts):.1f}")
        print(f"  Word count range: {min(word_counts)} - {max(word_counts)}")
    
    print("\n🧠 Complexity Metrics:")
    complexity_scores = [q['estimated_complexity_score'] for q in all_questions]
    if complexity_scores:
        print(f"  Average complexity: {sum(complexity_scores)/len(complexity_scores):.3f}")
        print(f"  Complexity range: {min(complexity_scores):.3f} - {max(complexity_scores):.3f}")
    
    print("\n🏷️ Tag Analysis:")
    all_tags = []
    for q in all_questions:
        all_tags.extend(q.get('tags', []))
    
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    # Show most common tags
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    print("  Most common tags:")
    for tag, count in sorted_tags[:10]:
        print(f"    {tag}: {count}")
    
    print("\n🌍 Domain Analysis:")
    domains = [q.get('domain') for q in all_questions if q.get('domain')]
    domain_counts = {}
    for domain in domains:
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    if domain_counts:
        print("  Domain distribution:")
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {domain}: {count}")
    else:
        print("  No domain information available")
    
    print("\n🔬 Knowledge Requirements:")
    specific_knowledge_count = sum(1 for q in all_questions if q.get('has_specific_knowledge', False))
    print(f"  Questions requiring specific knowledge: {specific_knowledge_count}/{len(all_questions)} "
          f"({specific_knowledge_count/len(all_questions)*100:.1f}%)")
    
    reasoning_steps = [q.get('requires_reasoning_steps', 0) for q in all_questions]
    if reasoning_steps:
        avg_reasoning = sum(reasoning_steps) / len(reasoning_steps)
        print(f"  Average reasoning steps required: {avg_reasoning:.1f}")


def demo_quality_validation():
    """Demonstrate quality validation features."""
    print("\n" + "="*60)
    print("🔍 DEMO: Quality Validation")
    print("="*60)
    
    from src.data_generation.enhanced_question_generator import QualityValidator
    
    validator = QualityValidator()
    
    # Test questions with different quality levels
    test_questions = [
        "What is the capital of France?",  # Good quality
        "Hi",  # Poor quality - too short
        "What are the underlying sociological and psychological factors that contribute to the phenomenon of urban gentrification, and how do these factors interact with economic policies to create displacement patterns that disproportionately affect marginalized communities?",  # High quality but long
        "This is not a question",  # Poor quality - no question mark
        "What what what what what?",  # Poor quality - repetitive
        "How do neural networks learn from data through backpropagation?",  # Good quality
        "",  # Poor quality - empty
        "Can you explain the relationship between quantum entanglement and information theory in the context of quantum computing algorithms?"  # High quality, complex
    ]
    
    print("🧪 Testing quality validation on sample questions:\n")
    
    for i, question in enumerate(test_questions, 1):
        if not question:
            display_question = "[EMPTY QUESTION]"
        else:
            display_question = question[:50] + "..." if len(question) > 50 else question
        
        is_valid, quality_score, issues = validator.validate_question(question)
        
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"{i}. {display_question}")
        print(f"   {status} | Quality Score: {quality_score:.3f}")
        
        if issues:
            print(f"   Issues: {', '.join(issues)}")
        print()


def main():
    """Run all demos."""
    print("🚀 NeuronMap Enhanced Question Generation Demo")
    print("=" * 60)
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    setup_logging()
    
    try:
        # Run demos
        demo_quality_validation()
        
        targeted_questions = demo_targeted_generation()
        
        balanced_questions = demo_balanced_generation()
        
        if targeted_questions or balanced_questions:
            demo_metadata_analysis()
        
        print("\n" + "="*60)
        print("🎉 Demo completed successfully!")
        print("="*60)
        print("\n📁 Generated files:")
        
        for file_path in ["data/demo_targeted_questions.json", "data/demo_balanced_questions.json"]:
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                print(f"  {file_path} ({file_size:,} bytes)")
        
        print("\n💡 Next steps:")
        print("  1. Examine the generated JSON files to see the rich metadata")
        print("  2. Use the enhanced questions for activation extraction")
        print("  3. Analyze how difficulty and category affect neural activations")
        print("  4. Experiment with different configuration settings")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()