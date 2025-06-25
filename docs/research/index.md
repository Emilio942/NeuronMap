# Research Methodology Guide

This guide provides comprehensive guidelines for conducting rigorous, reproducible research using NeuronMap.

## 🔬 Research Philosophy

NeuronMap is designed with scientific rigor in mind. This guide ensures that your neural network analysis meets the highest standards of research quality, reproducibility, and statistical validity.

### Core Principles

1. **Reproducibility**: All experiments must be fully reproducible
2. **Statistical Rigor**: Proper statistical methods and power analysis
3. **Transparency**: Clear documentation of methods and assumptions
4. **Validation**: Independent validation of results
5. **Open Science**: Shareable protocols and data

## 📋 Experimental Design Framework

### 1. Research Question Formulation

Before starting any analysis, clearly define your research questions:

```{admonition} Research Question Template
:class: tip

**Primary Question**: What specific aspect of neural network behavior are you investigating?

**Secondary Questions**: 
- How does this behavior vary across models?
- What factors influence this behavior?
- How generalizable are the findings?

**Hypotheses**: 
- H₁: Clear, testable hypothesis
- H₀: Null hypothesis
- Alternative hypotheses

**Success Criteria**: Specific, measurable outcomes that would support your hypothesis
```

### 2. Experimental Design Types

#### A. Comparative Studies

Comparing activation patterns across different models or conditions:

```python
from src.research.experimental_design import ComparativeStudy
from src.utils.config import get_config_manager

# Define experimental parameters
study = ComparativeStudy(
    name="cross_model_comparison",
    models=["gpt2", "bert-base", "distilbert"],
    conditions=["factual", "creative", "reasoning"],
    sample_size_per_condition=100,
    randomization_seed=42
)

# Power analysis
power_analysis = study.calculate_power(
    effect_size=0.5,  # Cohen's d
    alpha=0.05,
    power=0.8
)

print(f"Required sample size: {power_analysis.required_n}")
```

#### B. Intervention Studies

Testing the effect of specific interventions:

```python
from src.research.experimental_design import InterventionStudy

study = InterventionStudy(
    name="prompt_engineering_effects",
    baseline_condition="standard_prompts",
    intervention_conditions=[
        "chain_of_thought",
        "few_shot_examples", 
        "systematic_prompting"
    ],
    control_variables=["model_size", "domain", "complexity"],
    blocking_factors=["question_category"]
)

# Randomization scheme
randomization = study.generate_randomization_scheme(
    total_samples=1000,
    stratify_by=["question_category", "complexity"]
)
```

#### C. Longitudinal Studies

Tracking changes over time or training steps:

```python
from src.research.experimental_design import LongitudinalStudy

study = LongitudinalStudy(
    name="training_dynamics",
    time_points=[0, 1000, 5000, 10000, 50000],  # Training steps
    measures=["activation_magnitude", "attention_entropy", "layer_similarity"],
    subjects=["model_checkpoint_{}".format(i) for i in range(10)],
    repeated_measures=True
)

# Time series analysis plan
analysis_plan = study.create_analysis_plan(
    primary_outcome="activation_magnitude",
    covariates=["training_step", "data_complexity"],
    random_effects=["model_instance"]
)
```

### 3. Sample Size Calculation

Always perform power analysis before data collection:

```python
from src.research.statistics import PowerAnalysis

# For comparing means between two groups
power_calc = PowerAnalysis()

sample_size = power_calc.two_sample_t_test(
    effect_size=0.5,      # Cohen's d
    alpha=0.05,           # Type I error rate
    power=0.8,            # Desired power
    two_tailed=True
)

print(f"Required sample size per group: {sample_size}")

# For correlation analysis
correlation_sample = power_calc.correlation_analysis(
    expected_correlation=0.3,
    alpha=0.05,
    power=0.8
)

print(f"Required sample size for correlation: {correlation_sample}")
```

## 🧪 Experimental Protocols

### Protocol 1: Activation Pattern Analysis

**Objective**: Compare activation patterns across different question types

```python
# 1. Setup and Configuration
from src.research.protocols import ActivationPatternProtocol
from src.utils.config import setup_global_config

config = setup_global_config(environment="research")
protocol = ActivationPatternProtocol(config=config)

# 2. Define experimental parameters
parameters = {
    "models": ["gpt2", "gpt2-medium", "gpt2-large"],
    "question_types": ["factual", "reasoning", "creative", "ethical"],
    "layers_of_interest": [3, 6, 9, 12],  # 25%, 50%, 75%, 100% depth
    "sample_size_per_condition": 50,
    "randomization_seed": 42
}

# 3. Generate stimuli
stimuli = protocol.generate_balanced_stimuli(
    n_per_category=parameters["sample_size_per_condition"],
    categories=parameters["question_types"],
    complexity_levels=[1, 2, 3],  # Low, medium, high
    randomize=True,
    seed=parameters["randomization_seed"]
)

# 4. Run experiment
results = protocol.run_experiment(
    stimuli=stimuli,
    models=parameters["models"],
    layers=parameters["layers_of_interest"],
    batch_size=config.get_analysis_config().batch_size
)

# 5. Statistical analysis
analysis = protocol.analyze_results(
    results=results,
    primary_dv="activation_magnitude",
    factors=["model", "question_type", "layer"],
    covariates=["question_length", "complexity"]
)

# 6. Generate report
report = protocol.generate_report(analysis)
```

### Protocol 2: Attention Head Specialization

**Objective**: Identify specialized attention heads across models

```python
from src.research.protocols import AttentionSpecializationProtocol

protocol = AttentionSpecializationProtocol()

# Experimental design
design = {
    "models": ["bert-base", "roberta-base", "distilbert"],
    "task_types": ["syntax", "semantics", "pragmatics"],
    "linguistic_phenomena": [
        "subject_verb_agreement",
        "coreference_resolution", 
        "semantic_similarity",
        "syntactic_parsing"
    ]
}

# Run attention analysis
attention_results = protocol.run_attention_analysis(
    design=design,
    attention_metrics=[
        "head_entropy",
        "attention_distance", 
        "specialization_index",
        "consistency_score"
    ]
)

# Statistical modeling
specialization_model = protocol.fit_specialization_model(
    data=attention_results,
    predictors=["model", "layer", "head", "phenomenon"],
    outcome="specialization_index"
)
```

### Protocol 3: Cross-Model Generalization

**Objective**: Test whether findings generalize across model architectures

```python
from src.research.protocols import GeneralizationProtocol

protocol = GeneralizationProtocol()

# Multi-architecture comparison
architectures = {
    "encoder_only": ["bert-base", "roberta-base", "distilbert"],
    "decoder_only": ["gpt2", "gpt2-medium", "distilgpt2"],
    "encoder_decoder": ["t5-small", "t5-base", "bart-base"]
}

# Cross-validation design
cv_design = protocol.create_cross_validation_design(
    architectures=architectures,
    n_folds=5,
    stratify_by=["architecture_type", "model_size"],
    test_phenomena=["activation_clustering", "layer_similarity"]
)

# Run generalization study
generalization_results = protocol.run_generalization_study(
    cv_design=cv_design,
    metrics=["silhouette_score", "adjusted_rand_index", "homogeneity_score"]
)
```

## 📊 Statistical Analysis Guidelines

### 1. Descriptive Statistics

Always start with comprehensive descriptive analysis:

```python
from src.research.statistics import DescriptiveAnalysis

desc_analysis = DescriptiveAnalysis()

# Basic descriptives
descriptives = desc_analysis.compute_descriptives(
    data=activation_data,
    grouping_vars=["model", "layer", "question_type"],
    measures=["mean", "std", "median", "iqr", "skewness", "kurtosis"]
)

# Effect size calculations
effect_sizes = desc_analysis.compute_effect_sizes(
    data=activation_data,
    groups=["experimental", "control"],
    effect_types=["cohens_d", "eta_squared", "omega_squared"]
)

# Confidence intervals
confidence_intervals = desc_analysis.bootstrap_confidence_intervals(
    data=activation_data,
    statistic="mean",
    n_bootstrap=10000,
    confidence_level=0.95
)
```

### 2. Inferential Statistics

Choose appropriate statistical tests based on your data and design:

```python
from src.research.statistics import InferentialAnalysis

inferential = InferentialAnalysis()

# For comparing means across groups
if design_type == "between_subjects":
    results = inferential.one_way_anova(
        data=activation_data,
        dv="activation_magnitude",
        iv="condition",
        post_hoc="tukey"
    )
elif design_type == "mixed_design":
    results = inferential.mixed_anova(
        data=activation_data,
        dv="activation_magnitude",
        between="model",
        within="layer",
        subject="participant_id"
    )

# For correlation analysis
correlation_matrix = inferential.correlation_analysis(
    data=activation_data,
    variables=["layer_1", "layer_2", "layer_3"],
    method="pearson",
    correction="bonferroni"
)

# For multivariate analysis
manova_results = inferential.manova(
    data=activation_data,
    dvs=["activation_magnitude", "attention_entropy"],
    ivs=["model", "condition"],
    interaction_terms=["model*condition"]
)
```

### 3. Multiple Comparisons Correction

When performing multiple tests, always correct for multiple comparisons:

```python
from src.research.statistics import MultipleComparisons

mc = MultipleComparisons()

# Family-wise error rate control
corrected_p_values = mc.correct_p_values(
    p_values=raw_p_values,
    method="holm",  # Options: "bonferroni", "holm", "hochberg", "bh", "by"
    alpha=0.05
)

# False discovery rate control  
fdr_results = mc.false_discovery_rate(
    p_values=raw_p_values,
    method="benjamini_hochberg",
    alpha=0.05
)

# Report corrected results
mc.generate_correction_report(
    original_p=raw_p_values,
    corrected_p=corrected_p_values,
    method="holm",
    alpha=0.05
)
```

### 4. Model Validation

Validate your statistical models:

```python
from src.research.statistics import ModelValidation

validator = ModelValidation()

# Assumption checking
assumptions = validator.check_assumptions(
    model=your_statistical_model,
    tests=["normality", "homoscedasticity", "independence", "linearity"]
)

# Cross-validation
cv_results = validator.cross_validate(
    model=your_model,
    data=your_data,
    cv_type="k_fold",
    k=10,
    metrics=["mse", "r_squared", "mae"]
)

# Model comparison
comparison = validator.compare_models(
    models=[model1, model2, model3],
    data=your_data,
    criteria=["aic", "bic", "likelihood_ratio"]
)
```

## 🔄 Reproducibility Framework

### 1. Version Control and Documentation

```python
from src.research.reproducibility import ExperimentTracker

tracker = ExperimentTracker()

# Initialize experiment
experiment = tracker.create_experiment(
    name="activation_pattern_analysis",
    description="Comparing activation patterns across transformer models",
    hypothesis="Different model architectures show distinct activation patterns",
    researcher="Your Name",
    institution="Your Institution"
)

# Log experiment parameters
experiment.log_parameters({
    "models": ["gpt2", "bert-base"],
    "sample_size": 100,
    "random_seed": 42,
    "analysis_method": "mixed_anova"
})

# Log code version
experiment.log_code_version(
    repository="https://github.com/Emilio942/NeuronMap",
    commit_hash="abc123def456",
    branch="main"
)

# Log environment
experiment.log_environment(
    python_version="3.9.7",
    cuda_version="11.2",
    package_versions=tracker.get_package_versions()
)
```

### 2. Data Management

```python
from src.research.reproducibility import DataManager

data_manager = DataManager()

# Create data manifest
manifest = data_manager.create_manifest(
    data_files=["stimuli.json", "responses.csv", "activations.h5"],
    checksums="sha256",
    metadata={
        "collection_date": "2025-06-23",
        "collection_method": "automated",
        "quality_checks": ["completeness", "format_validation"]
    }
)

# Validate data integrity
validation_report = data_manager.validate_data(
    files=["stimuli.json", "responses.csv"],
    expected_checksums=manifest.checksums,
    quality_checks=["missing_values", "outliers", "format_consistency"]
)

# Archive data
archive_path = data_manager.archive_data(
    experiment_id=experiment.id,
    data_files=["stimuli.json", "responses.csv", "activations.h5"],
    compression="gzip",
    encryption=True
)
```

### 3. Results Validation

```python
from src.research.reproducibility import ResultsValidator

validator = ResultsValidator()

# Independent validation
validation_results = validator.independent_validation(
    original_experiment=experiment,
    validation_data="validation_dataset.csv",
    replication_seed=43  # Different from original
)

# Cross-platform validation
platform_validation = validator.cross_platform_validation(
    experiment=experiment,
    platforms=["linux", "windows", "macos"],
    python_versions=["3.8", "3.9", "3.10"]
)

# Statistical replication
replication_report = validator.statistical_replication(
    original_results=original_results,
    replication_results=validation_results,
    significance_level=0.05,
    equivalence_bounds=[-0.1, 0.1]
)
```

## 📈 Quality Metrics

### 1. Experimental Quality Checklist

```python
from src.research.quality import ExperimentalQualityChecker

quality_checker = ExperimentalQualityChecker()

# Comprehensive quality assessment
quality_report = quality_checker.assess_experiment(
    experiment=experiment,
    criteria=[
        "sample_size_adequacy",
        "randomization_quality", 
        "control_for_confounds",
        "statistical_power",
        "effect_size_reporting",
        "confidence_intervals",
        "replication_plan"
    ]
)

# Generate quality score
quality_score = quality_checker.calculate_quality_score(
    experiment=experiment,
    weights={
        "methodological_rigor": 0.3,
        "statistical_validity": 0.3,
        "reproducibility": 0.2,
        "transparency": 0.2
    }
)

print(f"Experiment Quality Score: {quality_score:.2f}/10")
```

### 2. Publication Readiness

```python
from src.research.quality import PublicationChecker

pub_checker = PublicationChecker()

# Check publication standards
publication_report = pub_checker.check_publication_standards(
    experiment=experiment,
    standards=["consort", "arrive", "prisma"],  # Choose relevant standard
    required_sections=[
        "methods",
        "results", 
        "statistical_analysis",
        "data_availability",
        "code_availability"
    ]
)

# Generate publication-ready summary
summary = pub_checker.generate_publication_summary(
    experiment=experiment,
    format="apa",  # or "nature", "science", etc.
    include_figures=True,
    include_tables=True
)
```

## 🏆 Best Practices

### 1. Pre-registration

Always pre-register your analysis plan:

```python
from src.research.preregistration import PreregistrationTemplate

prereg = PreregistrationTemplate()

# Create pre-registration
registration = prereg.create_preregistration(
    title="Activation Pattern Analysis Across Transformer Models",
    researchers=["Your Name"],
    institution="Your Institution",
    research_questions=[
        "Do different transformer architectures show distinct activation patterns?",
        "How do activation patterns vary across layers within architectures?"
    ],
    hypotheses=[
        "H1: Encoder-only models show more uniform activation patterns",
        "H2: Decoder-only models show increasing specialization in deeper layers"
    ],
    design_description="Between-subjects design comparing activation patterns...",
    analysis_plan="Mixed ANOVA with architecture as between-subjects factor...",
    exclusion_criteria="Models with fewer than 6 layers will be excluded...",
    statistical_power="Power analysis indicates n=100 per condition for 80% power..."
)

# Submit pre-registration
registration_id = prereg.submit_preregistration(
    registration=registration,
    platform="osf",  # or other preregistration platform
    public=True
)
```

### 2. Robustness Checks

Always perform robustness checks:

```python
from src.research.robustness import RobustnessChecker

robustness = RobustnessChecker()

# Sensitivity analysis
sensitivity_results = robustness.sensitivity_analysis(
    original_analysis=your_analysis,
    parameter_variations={
        "alpha_level": [0.01, 0.05, 0.10],
        "outlier_threshold": [1.5, 2.0, 2.5, 3.0],
        "sample_size": [0.8, 0.9, 1.0, 1.1, 1.2]
    }
)

# Bootstrap analysis
bootstrap_results = robustness.bootstrap_analysis(
    data=your_data,
    analysis_function=your_analysis_function,
    n_bootstrap=10000,
    confidence_levels=[0.90, 0.95, 0.99]
)

# Subgroup analysis
subgroup_results = robustness.subgroup_analysis(
    data=your_data,
    subgroups=["model_size", "architecture_type"],
    analysis_function=your_analysis_function
)
```

### 3. Transparency and Sharing

```python
from src.research.sharing import DataSharing, CodeSharing

# Prepare data for sharing
data_sharer = DataSharing()
shared_data = data_sharer.prepare_for_sharing(
    data=your_data,
    anonymization=True,
    privacy_level="high",
    format="csv",
    documentation="full"
)

# Prepare code for sharing
code_sharer = CodeSharing()
shared_code = code_sharer.prepare_code_archive(
    experiment=experiment,
    include_dependencies=True,
    include_environment=True,
    documentation_level="comprehensive"
)

# Create research compendium
compendium = data_sharer.create_research_compendium(
    experiment=experiment,
    data=shared_data,
    code=shared_code,
    documentation="README.md",
    license="MIT"
)
```

## 📚 Templates and Checklists

### Research Protocol Template

```markdown
# Research Protocol: [Title]

## 1. Background and Rationale
- Research question
- Theoretical framework
- Literature review
- Gaps in knowledge

## 2. Objectives and Hypotheses
- Primary objective
- Secondary objectives
- Primary hypothesis
- Secondary hypotheses

## 3. Methods
- Study design
- Participants/Models
- Stimuli/Materials
- Procedure
- Data collection

## 4. Statistical Analysis Plan
- Primary analysis
- Secondary analyses
- Handling of missing data
- Multiple comparisons correction
- Sensitivity analyses

## 5. Quality Control
- Data validation procedures
- Code review process
- Independent verification
- Replication plan

## 6. Ethics and Transparency
- Ethical considerations
- Data sharing plan
- Code availability
- Preregistration details
```

### Quality Checklist

```{admonition} Experimental Quality Checklist
:class: important

**Design Quality**
- [ ] Clear research question and hypotheses
- [ ] Appropriate study design for research question
- [ ] Adequate sample size (power analysis performed)
- [ ] Proper randomization and controls
- [ ] Potential confounds identified and controlled

**Statistical Quality**
- [ ] Appropriate statistical methods chosen
- [ ] Assumptions checked and met
- [ ] Multiple comparisons corrected
- [ ] Effect sizes reported with confidence intervals
- [ ] Statistical significance appropriately interpreted

**Reproducibility**
- [ ] Complete analysis code available
- [ ] All dependencies documented
- [ ] Random seeds set and documented
- [ ] Data processing steps clearly documented
- [ ] Results independently verified

**Transparency**
- [ ] Pre-registration completed (if applicable)
- [ ] All analyses reported (including failed analyses)
- [ ] Data and code sharing plan implemented
- [ ] Clear documentation of all procedures
- [ ] Limitations and assumptions clearly stated
```

---

```{seealso}
- {doc}`experimental_design` - Detailed experimental design patterns
- {doc}`reproducibility` - Comprehensive reproducibility framework
- {doc}`../api/index` - API reference for research tools
- {doc}`../examples/research_examples` - Complete research examples
```
