"""
Domain Specialists Package
=========================

This package contains specialized question generators for different knowledge domains.
Each specialist implements domain-specific vocabulary, question patterns, and validation logic.
"""

# Import from the main domain_specialists.py file
try:
    # Import from parent directory
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

    from domain_specialists import (
        # Core classes
        DomainSpecialistFactory as DomainSpecializationFramework,  # Use factory as framework
        DomainSpecialist,
        DomainType,
        QuestionType,

        # Helper functions
        create_domain_specialist,
        get_available_domains,
        generate_domain_specific_questions,

        # All specialist classes that exist
        PhysicsSpecialist,
        ChemistrySpecialist,
        BiologySpecialist,
        MathematicsSpecialist,
        ComputerScienceSpecialist,
        EngineeringSpecialist,
        HistorySpecialist,
        PhilosophySpecialist,
        LiteratureSpecialist,
        LinguisticsSpecialist,
        ArtHistorySpecialist,
        PsychologySpecialist,
        SociologySpecialist,
        PoliticalScienceSpecialist,
        EconomicsSpecialist,
        AnthropologySpecialist,
        MedicineSpecialist,
        LawSpecialist,
        EducationSpecialist,
        BusinessSpecialist
    )

    # Make everything available at package level
    __all__ = [
        'DomainSpecializationFramework',
        'DomainSpecialist',
        'DomainType',
        'QuestionType',
        'create_domain_specialist',
        'get_available_domains',
        'generate_domain_specific_questions',
        'PhysicsSpecialist',
        'ChemistrySpecialist',
        'BiologySpecialist',
        'MathematicsSpecialist',
        'ComputerScienceSpecialist',
        'EngineeringSpecialist',
        'HistorySpecialist',
        'PhilosophySpecialist',
        'LiteratureSpecialist',
        'LinguisticsSpecialist',
        'ArtHistorySpecialist',
        'PsychologySpecialist',
        'SociologySpecialist',
        'PoliticalScienceSpecialist',
        'EconomicsSpecialist',
        'AnthropologySpecialist',
        'MedicineSpecialist',
        'LawSpecialist',
        'EducationSpecialist',
        'BusinessSpecialist'
    ]

except ImportError as e:
    print(f"Warning: Could not import from domain_specialists.py: {e}")
    # Set fallback exports
    __all__ = []
