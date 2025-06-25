# Experimental Design Patterns

This guide provides specific experimental design patterns commonly used in neural network analysis research with NeuronMap.

## 🧪 Core Design Patterns

### Pattern 1: Cross-Architecture Comparison

**Use Case**: Comparing activation patterns across different model architectures

```python
from src.research.patterns import CrossArchitecturePattern
from src.utils.config import get_config_manager

# Initialize pattern
pattern = CrossArchitecturePattern()

# Define architecture groups
architectures = {
    "encoder_only": ["bert-base", "roberta-base", "distilbert-base"],
    "decoder_only": ["gpt2", "gpt2-medium", "distilgpt2"],
    "encoder_decoder": ["t5-small", "t5-base", "bart-base"]
}

# Design experimental matrix
design_matrix = pattern.create_design_matrix(
    architectures=architectures,
    factors=["model_size", "training_data", "architecture_type"],
    balanced=True,
    randomize=True
)

# Generate analysis plan
analysis_plan = pattern.create_analysis_plan(
    primary_dv="activation_similarity",
    secondary_dvs=["attention_entropy", "layer_specialization"],
    between_factors=["architecture_type", "model_size"],
    statistical_tests=["mixed_anova", "post_hoc_tukey"]
)

# Execute experiment
results = pattern.execute_experiment(
    design_matrix=design_matrix,
    stimuli=question_stimuli,
    analysis_plan=analysis_plan
)
```

### Pattern 2: Layer-wise Developmental Analysis

**Use Case**: Tracking how representations develop across layers

```python
from src.research.patterns import LayerwisePattern

pattern = LayerwisePattern()

# Define layer analysis parameters
layer_config = {
    "models": ["gpt2", "bert-base"],
    "layer_percentages": [0.2, 0.4, 0.6, 0.8, 1.0],  # 20%, 40%, etc.
    "measures": [
        "activation_magnitude",
        "representation_similarity", 
        "attention_concentration",
        "gradient_flow"
    ]
}

# Create developmental trajectory analysis
trajectory_design = pattern.create_trajectory_design(
    layer_config=layer_config,
    trajectory_type="polynomial",  # linear, polynomial, spline
    time_structure="ordered",      # ordered, unordered
    repeated_measures=True
)

# Fit developmental models
developmental_models = pattern.fit_developmental_models(
    data=activation_data,
    trajectories=["linear", "quadratic", "cubic"],
    random_effects=["model_instance", "stimulus_category"]
)

# Compare developmental patterns
pattern_comparison = pattern.compare_developmental_patterns(
    models=developmental_models,
    comparison_metrics=["aic", "bic", "likelihood_ratio"],
    visualization=True
)
```

### Pattern 3: Intervention Study Design

**Use Case**: Testing the effect of specific interventions on neural representations

```python
from src.research.patterns import InterventionPattern

pattern = InterventionPattern()

# Define intervention parameters
intervention_config = {
    "baseline": "standard_prompts",
    "interventions": [
        {
            "name": "chain_of_thought",
            "description": "Add 'Let's think step by step' to prompts",
            "implementation": lambda x: f"Let's think step by step. {x}"
        },
        {
            "name": "few_shot",
            "description": "Provide 3 examples before each question",
            "implementation": lambda x: f"{few_shot_examples}\n\nQuestion: {x}"
        },
        {
            "name": "systematic_prompting",
            "description": "Use structured prompting template",
            "implementation": lambda x: f"Problem: {x}\nApproach: [think systematically]\nSolution:"
        }
    ],
    "control_variables": ["question_difficulty", "domain", "length"]
}

# Create intervention design
intervention_design = pattern.create_intervention_design(
    config=intervention_config,
    design_type="randomized_controlled",  # rct, crossover, factorial
    allocation_ratio="1:1:1:1",  # baseline:intervention1:intervention2:intervention3
    stratification_vars=["question_difficulty", "domain"]
)

# Randomization scheme
randomization = pattern.generate_randomization(
    n_total=400,
    block_size=4,
    stratification=intervention_design.stratification_vars,
    seed=42
)

# Analysis plan for interventions
intervention_analysis = pattern.create_intervention_analysis_plan(
    primary_outcome="activation_change",
    secondary_outcomes=["attention_shift", "layer_activation"],
    covariates=["baseline_activation", "question_complexity"],
    statistical_approach="ancova"  # ancova, mixed_model, difference_in_differences
)
```

### Pattern 4: Longitudinal Training Dynamics

**Use Case**: Studying how representations change during model training

```python
from src.research.patterns import LongitudinalPattern

pattern = LongitudinalPattern()

# Define longitudinal parameters
longitudinal_config = {
    "checkpoints": [0, 1000, 5000, 10000, 25000, 50000, 100000],  # Training steps
    "models": ["gpt2_checkpoint_{step}" for step in [0, 1000, 5000, 10000, 25000, 50000, 100000]],
    "tracking_measures": [
        "activation_variance",
        "layer_similarity",
        "attention_patterns",
        "representation_geometry"
    ],
    "stability_metrics": [
        "representation_drift",
        "catastrophic_forgetting",
        "progressive_refinement"
    ]
}

# Create longitudinal design
longitudinal_design = pattern.create_longitudinal_design(
    config=longitudinal_config,
    design_type="cohort",  # cohort, panel, time_series
    measurement_schedule="fixed_intervals",
    missing_data_strategy="multiple_imputation"
)

# Time series analysis
time_series_analysis = pattern.setup_time_series_analysis(
    data=training_data,
    time_var="training_step",
    outcome_vars=longitudinal_config["tracking_measures"],
    trend_analysis=["linear", "polynomial", "changepoint"],
    seasonality=False,  # No seasonality in training
    autocorrelation_structure="ar1"  # AR(1) model for correlation
)

# Growth curve modeling
growth_models = pattern.fit_growth_models(
    data=training_data,
    time_points=longitudinal_config["checkpoints"],
    outcomes=longitudinal_config["tracking_measures"],
    model_types=["linear", "quadratic", "piecewise"],
    random_effects=["model_initialization", "data_order"]
)
```

### Pattern 5: Multi-factorial Design

**Use Case**: Examining multiple factors simultaneously

```python
from src.research.patterns import MultifactorialPattern

pattern = MultifactorialPattern()

# Define factorial structure
factorial_config = {
    "factors": {
        "model_type": ["gpt", "bert", "t5"],           # 3 levels
        "model_size": ["small", "medium", "large"],    # 3 levels  
        "question_type": ["factual", "reasoning"],     # 2 levels
        "complexity": ["low", "high"]                  # 2 levels
    },
    "design_type": "full_factorial",  # full_factorial, fractional_factorial
    "replication": 10,  # Number of replications per cell
    "blocking": "question_category"  # Blocking factor
}

# Create factorial design
factorial_design = pattern.create_factorial_design(
    config=factorial_config,
    randomization="complete",  # complete, restricted, blocked
    counterbalancing=True
)

# Power analysis for factorial design
power_analysis = pattern.factorial_power_analysis(
    design=factorial_design,
    effect_sizes={
        "main_effects": {"model_type": 0.5, "model_size": 0.3, "question_type": 0.4},
        "interactions": {"model_type*question_type": 0.2}
    },
    alpha=0.05,
    desired_power=0.8
)

# Statistical analysis plan
factorial_analysis = pattern.create_factorial_analysis_plan(
    design=factorial_design,
    primary_effects=["model_type", "question_type"],
    interaction_effects=["model_type*question_type", "model_size*complexity"],
    post_hoc_comparisons="tukey",
    effect_size_measures=["eta_squared", "partial_eta_squared"]
)
```

### Pattern 6: Equivalence Testing

**Use Case**: Testing whether two models perform equivalently

```python
from src.research.patterns import EquivalencePattern

pattern = EquivalencePattern()

# Define equivalence parameters
equivalence_config = {
    "models": ["model_a", "model_b"],
    "equivalence_margin": 0.1,  # ±0.1 standardized units
    "alpha": 0.05,
    "power": 0.8,
    "equivalence_type": "two_one_sided"  # two_one_sided, interval
}

# Sample size for equivalence testing
equivalence_sample = pattern.calculate_equivalence_sample_size(
    config=equivalence_config,
    expected_difference=0.05,  # Expected small difference
    variance_estimate=1.0
)

# Equivalence testing procedure
equivalence_test = pattern.conduct_equivalence_test(
    data_a=model_a_activations,
    data_b=model_b_activations,
    equivalence_margin=equivalence_config["equivalence_margin"],
    test_type="two_one_sided_t_test"
)

# Confidence interval approach
confidence_interval = pattern.equivalence_confidence_interval(
    data_a=model_a_activations,
    data_b=model_b_activations,
    confidence_level=0.90,  # (1-2*alpha) for equivalence testing
    equivalence_bounds=[-0.1, 0.1]
)
```

## 🎯 Design Selection Guide

### Choosing the Right Design

```python
from src.research.patterns import DesignSelector

selector = DesignSelector()

# Get design recommendation
recommendation = selector.recommend_design(
    research_question="Do different model architectures process language differently?",
    variables={
        "independent": ["model_architecture", "layer_depth"],
        "dependent": ["activation_pattern"],
        "confounding": ["model_size", "training_data"]
    },
    constraints={
        "computational_budget": "medium",
        "time_constraint": "6_weeks", 
        "sample_availability": "unlimited"
    },
    research_goals=["causal_inference", "generalizability"]
)

print(f"Recommended design: {recommendation.design_type}")
print(f"Justification: {recommendation.justification}")
print(f"Estimated sample size: {recommendation.sample_size}")
print(f"Expected power: {recommendation.power}")
```

### Design Optimization

```python
from src.research.patterns import DesignOptimizer

optimizer = DesignOptimizer()

# Optimize design parameters
optimized_design = optimizer.optimize_design(
    base_design=your_initial_design,
    optimization_criteria=[
        "maximize_power",
        "minimize_sample_size",
        "control_type_i_error"
    ],
    constraints={
        "max_sample_size": 1000,
        "min_power": 0.8,
        "max_alpha": 0.05,
        "computational_limit": "high"
    }
)

# Sensitivity analysis
sensitivity_analysis = optimizer.design_sensitivity_analysis(
    design=optimized_design,
    parameter_variations={
        "effect_size": [0.2, 0.3, 0.4, 0.5],
        "sample_size": [50, 100, 200, 500],
        "alpha_level": [0.01, 0.05, 0.10]
    }
)
```

## 📊 Analysis Templates

### Template 1: Mixed-Effects Analysis

```python
from src.research.analysis_templates import MixedEffectsTemplate

template = MixedEffectsTemplate()

# Setup mixed effects model
model_spec = template.create_model_specification(
    outcome="activation_magnitude",
    fixed_effects=["model_type", "layer", "question_complexity"],
    random_effects=["subject_id", "stimulus_id"],
    interactions=["model_type*layer"],
    covariates=["baseline_activation"]
)

# Fit model with proper contrasts
model_results = template.fit_mixed_model(
    data=your_data,
    model_spec=model_spec,
    contrasts="treatment",  # treatment, sum, helmert
    estimation_method="reml"  # ml, reml
)

# Model diagnostics
diagnostics = template.model_diagnostics(
    model=model_results,
    checks=["residual_normality", "homoscedasticity", "linearity", "leverage"]
)

# Post-hoc analysis
posthoc_results = template.posthoc_analysis(
    model=model_results,
    comparisons="all_pairwise",
    correction="tukey",
    effect_size="cohens_d"
)
```

### Template 2: Time Series Analysis

```python
from src.research.analysis_templates import TimeSeriesTemplate

template = TimeSeriesTemplate()

# Prepare time series data
ts_data = template.prepare_time_series(
    data=longitudinal_data,
    time_var="training_step",
    outcome_var="representation_similarity",
    grouping_var="model_id"
)

# Trend analysis
trend_analysis = template.trend_analysis(
    data=ts_data,
    trend_types=["linear", "polynomial", "spline"],
    change_point_detection=True,
    seasonal_components=False
)

# Autocorrelation analysis
autocorr_analysis = template.autocorrelation_analysis(
    data=ts_data,
    max_lag=20,
    significance_level=0.05,
    partial_autocorr=True
)

# Forecasting
forecast_results = template.forecast_analysis(
    data=ts_data,
    forecast_horizon=10,
    model_types=["arima", "exponential_smoothing", "neural_network"],
    validation_method="time_series_split"
)
```

### Template 3: Multivariate Analysis

```python
from src.research.analysis_templates import MultivariateTemplate

template = MultivariateTemplate()

# MANOVA setup
manova_spec = template.create_manova_specification(
    dependent_vars=["activation_mag", "attention_entropy", "layer_similarity"],
    independent_vars=["model_type", "stimulus_category"],
    covariates=["stimulus_length", "complexity_score"]
)

# Fit MANOVA
manova_results = template.fit_manova(
    data=your_data,
    specification=manova_spec,
    test_statistics=["wilks", "pillai", "hotelling", "roy"]
)

# Discriminant analysis
discriminant_analysis = template.discriminant_analysis(
    data=your_data,
    grouping_var="model_type",
    predictor_vars=["activation_mag", "attention_entropy"],
    cross_validation=True
)

# Principal component analysis
pca_results = template.principal_component_analysis(
    data=your_data,
    variables=["layer_1", "layer_2", "layer_3", "layer_4"],
    n_components="optimal",  # number or "optimal"
    rotation="varimax"
)
```

## 🔍 Design Validation

### Simulation Studies

```python
from src.research.validation import DesignSimulator

simulator = DesignSimulator()

# Simulate design performance
simulation_results = simulator.simulate_design(
    design=your_experimental_design,
    true_effects={
        "main_effect_model": 0.5,
        "main_effect_layer": 0.3,
        "interaction": 0.2
    },
    error_variance=1.0,
    n_simulations=1000,
    alpha=0.05
)

# Power analysis from simulations
power_analysis = simulator.empirical_power_analysis(
    simulation_results=simulation_results,
    effects_of_interest=["main_effect_model", "interaction"],
    alpha=0.05
)

# Type I error rate
type_i_error = simulator.type_i_error_analysis(
    design=your_experimental_design,
    null_effects=True,
    n_simulations=10000,
    alpha=0.05
)
```

### Cross-Validation Framework

```python
from src.research.validation import CrossValidationFramework

cv_framework = CrossValidationFramework()

# Design cross-validation
cv_results = cv_framework.cross_validate_design(
    design=your_design,
    data=your_data,
    cv_type="stratified_k_fold",
    k=10,
    metrics=["prediction_accuracy", "effect_size_consistency", "p_value_distribution"]
)

# External validation
external_validation = cv_framework.external_validation(
    training_design=your_design,
    training_data=training_data,
    external_data=external_data,
    validation_metrics=["effect_size_replication", "direction_consistency"]
)
```

---

```{seealso}
- {doc}`index` - Main research methodology guide
- {doc}`reproducibility` - Reproducibility framework
- {doc}`../api/analysis` - Analysis API reference
- {doc}`../tutorials/research_tutorial` - Step-by-step research tutorial
```
