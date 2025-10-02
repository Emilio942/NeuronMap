"""Scientific rigor and statistical testing module.

This module implements statistical tests, experiment logging, and reproducibility features:
- Statistical significance tests for activation comparisons
- Multiple comparison corrections (Bonferroni, FDR)
- Cross-validation for robust results
- Effect size calculations
- Experiment logging and reproducibility tools
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import warnings

# Optional dependencies with fallbacks
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some statistical tests will be unavailable.")

try:
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "Scikit-learn not available. Some statistical tests will be unavailable.")

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Results from a statistical test."""
    test_name: str
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    significant: bool
    alpha: float
    interpretation: str


@dataclass
class MultipleComparisonResult:
    """Results from multiple comparison correction."""
    original_p_values: List[float]
    corrected_p_values: List[float]
    significant_tests: List[bool]
    method: str
    alpha: float
    num_comparisons: int


@dataclass
class ExperimentMetadata:
    """Metadata for experiment reproducibility."""
    experiment_id: str
    timestamp: str
    model_name: str
    model_hash: str
    data_hash: str
    code_hash: str
    hyperparameters: Dict[str, Any]
    environment_info: Dict[str, str]
    random_seeds: Dict[str, int]
    results_summary: Dict[str, Any]


class StatisticalTester:
    """Statistical testing for neural network analysis."""

    def __init__(self, alpha: float = 0.05):
        """Initialize statistical tester.

        Args:
            alpha: Significance level for tests
        """
        self.alpha = alpha

    def compare_activations(self,
                            activations1: np.ndarray,
                            activations2: np.ndarray,
                            test_type: str = "wilcoxon") -> StatisticalTestResult:
        """Compare two sets of activations statistically.

        Args:
            activations1: First set of activations
            activations2: Second set of activations
            test_type: Type of test ("wilcoxon", "ttest", "permutation")

        Returns:
            StatisticalTestResult with test results
        """
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, returning mock results")
            return self._mock_test_result(test_type)

        # Flatten activations for comparison
        flat1 = activations1.flatten()
        flat2 = activations2.flatten()

        if test_type == "wilcoxon":
            return self._wilcoxon_test(flat1, flat2)
        elif test_type == "ttest":
            return self._t_test(flat1, flat2)
        elif test_type == "permutation":
            return self._permutation_test(flat1, flat2)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def _wilcoxon_test(
            self,
            data1: np.ndarray,
            data2: np.ndarray) -> StatisticalTestResult:
        """Perform Wilcoxon signed-rank test."""
        try:
            # Paired test if same length, otherwise Mann-Whitney U
            if len(data1) == len(data2):
                statistic, p_value = stats.wilcoxon(data1, data2)
                test_name = "Wilcoxon signed-rank test"
            else:
                statistic, p_value = stats.mannwhitneyu(data1, data2)
                test_name = "Mann-Whitney U test"

            # Calculate effect size (rank-biserial correlation)
            effect_size = self._calculate_rank_biserial_correlation(data1, data2)

            # Bootstrap confidence interval for effect size
            ci_lower, ci_upper = self._bootstrap_effect_size_ci(data1, data2)

            significant = p_value < self.alpha
            interpretation = self._interpret_wilcoxon_result(
                statistic, p_value, effect_size)

            return StatisticalTestResult(
                test_name=test_name,
                test_statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                significant=significant,
                alpha=self.alpha,
                interpretation=interpretation
            )

        except Exception as e:
            logger.error(f"Wilcoxon test failed: {e}")
            return self._mock_test_result("wilcoxon")

    def _t_test(self, data1: np.ndarray, data2: np.ndarray) -> StatisticalTestResult:
        """Perform t-test."""
        try:
            # Check for equal variances
            levene_stat, levene_p = stats.levene(data1, data2)
            equal_var = levene_p > 0.05

            # Perform appropriate t-test
            if len(data1) == len(data2):
                statistic, p_value = stats.ttest_rel(data1, data2)
                test_name = "Paired t-test"
            else:
                statistic, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
                test_name = f"Independent t-test ({
                    'equal' if equal_var else 'unequal'} variance)"

            # Calculate Cohen's d effect size
            effect_size = self._calculate_cohens_d(data1, data2)

            # Confidence interval for effect size
            ci_lower, ci_upper = self._bootstrap_effect_size_ci(data1, data2)

            significant = p_value < self.alpha
            interpretation = self._interpret_t_test_result(
                statistic, p_value, effect_size)

            return StatisticalTestResult(
                test_name=test_name,
                test_statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                significant=significant,
                alpha=self.alpha,
                interpretation=interpretation
            )

        except Exception as e:
            logger.error(f"T-test failed: {e}")
            return self._mock_test_result("ttest")

    def _permutation_test(self, data1: np.ndarray, data2: np.ndarray,
                          n_permutations: int = 10000) -> StatisticalTestResult:
        """Perform permutation test."""
        try:
            # Calculate observed difference in means
            observed_diff = np.mean(data1) - np.mean(data2)

            # Combine data
            combined = np.concatenate([data1, data2])
            n1 = len(data1)

            # Generate permutation distribution
            perm_diffs = []
            for _ in range(n_permutations):
                # Randomly shuffle and split
                np.random.shuffle(combined)
                perm_data1 = combined[:n1]
                perm_data2 = combined[n1:]

                perm_diff = np.mean(perm_data1) - np.mean(perm_data2)
                perm_diffs.append(perm_diff)

            perm_diffs = np.array(perm_diffs)

            # Calculate p-value (two-tailed)
            p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

            # Effect size as standardized difference
            pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
            effect_size = observed_diff / pooled_std if pooled_std > 0 else 0

            # Confidence interval from permutation distribution
            ci_lower = np.percentile(perm_diffs, 2.5)
            ci_upper = np.percentile(perm_diffs, 97.5)

            significant = p_value < self.alpha
            interpretation = self._interpret_permutation_result(
                observed_diff, p_value, effect_size)

            return StatisticalTestResult(
                test_name="Permutation test",
                test_statistic=observed_diff,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                significant=significant,
                alpha=self.alpha,
                interpretation=interpretation
            )

        except Exception as e:
            logger.error(f"Permutation test failed: {e}")
            return self._mock_test_result("permutation")

    def _calculate_cohens_d(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        n1, n2 = len(data1), len(data2)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    def _calculate_rank_biserial_correlation(
            self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate rank-biserial correlation effect size."""
        if len(data1) != len(data2):
            # For Mann-Whitney U, use different formula
            n1, n2 = len(data1), len(data2)
            u_statistic, _ = stats.mannwhitneyu(data1, data2)
            return 2 * u_statistic / (n1 * n2) - 1
        else:
            # For Wilcoxon, use simpler calculation
            differences = data1 - data2
            return np.mean(np.sign(differences))

    def _bootstrap_effect_size_ci(self, data1: np.ndarray, data2: np.ndarray,
                                  n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate confidence interval for effect size using bootstrap."""
        effect_sizes = []

        for _ in range(n_bootstrap):
            # Bootstrap samples
            boot_data1 = np.random.choice(data1, size=len(data1), replace=True)
            boot_data2 = np.random.choice(data2, size=len(data2), replace=True)

            # Calculate effect size
            effect_size = self._calculate_cohens_d(boot_data1, boot_data2)
            effect_sizes.append(effect_size)

        return np.percentile(effect_sizes, [2.5, 97.5])

    def _interpret_wilcoxon_result(
            self,
            statistic: float,
            p_value: float,
            effect_size: float) -> str:
        """Interpret Wilcoxon test result."""
        significance = "significant" if p_value < self.alpha else "not significant"

        if abs(effect_size) < 0.1:
            magnitude = "negligible"
        elif abs(effect_size) < 0.3:
            magnitude = "small"
        elif abs(effect_size) < 0.5:
            magnitude = "medium"
        else:
            magnitude = "large"

        return f"The test is {significance} (p={p_value:.4f}) with {
            magnitude} effect size (r={effect_size:.3f})"

    def _interpret_t_test_result(
            self,
            statistic: float,
            p_value: float,
            effect_size: float) -> str:
        """Interpret t-test result."""
        significance = "significant" if p_value < self.alpha else "not significant"

        if abs(effect_size) < 0.2:
            magnitude = "small"
        elif abs(effect_size) < 0.5:
            magnitude = "medium"
        elif abs(effect_size) < 0.8:
            magnitude = "large"
        else:
            magnitude = "very large"

        return f"The test is {significance} (t={statistic:.3f}, p={p_value:.4f}) with {
            magnitude} effect size (d={effect_size:.3f})"

    def _interpret_permutation_result(
            self,
            statistic: float,
            p_value: float,
            effect_size: float) -> str:
        """Interpret permutation test result."""
        significance = "significant" if p_value < self.alpha else "not significant"

        return f"The permutation test is {significance} (observed difference={
            statistic:.4f}, p={p_value:.4f})"

    def _mock_test_result(self, test_type: str) -> StatisticalTestResult:
        """Create mock test result when dependencies are unavailable."""
        return StatisticalTestResult(
            test_name=f"Mock {test_type} test",
            test_statistic=0.0,
            p_value=0.5,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            significant=False,
            alpha=self.alpha,
            interpretation="Mock result - dependencies not available"
        )


class MultipleComparisonCorrector:
    """Multiple comparison correction methods."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def correct_p_values(self, p_values: List[float],
                         method: str = "fdr_bh") -> MultipleComparisonResult:
        """Apply multiple comparison correction.

        Args:
            p_values: List of p-values to correct
            method: Correction method ("bonferroni", "fdr_bh", "fdr_by")

        Returns:
            MultipleComparisonResult with corrected p-values
        """
        p_array = np.array(p_values)

        if method == "bonferroni":
            corrected_p = self._bonferroni_correction(p_array)
        elif method == "fdr_bh":
            corrected_p = self._fdr_bh_correction(p_array)
        elif method == "fdr_by":
            corrected_p = self._fdr_by_correction(p_array)
        else:
            raise ValueError(f"Unknown correction method: {method}")

        significant = corrected_p < self.alpha

        return MultipleComparisonResult(
            original_p_values=p_values,
            corrected_p_values=corrected_p.tolist(),
            significant_tests=significant.tolist(),
            method=method,
            alpha=self.alpha,
            num_comparisons=len(p_values)
        )

    def _bonferroni_correction(self, p_values: np.ndarray) -> np.ndarray:
        """Apply Bonferroni correction."""
        return np.minimum(p_values * len(p_values), 1.0)

    def _fdr_bh_correction(self, p_values: np.ndarray) -> np.ndarray:
        """Apply Benjamini-Hochberg FDR correction."""
        if SCIPY_AVAILABLE:
            try:
                return false_discovery_control(p_values, method='bh')
            except BaseException:
                pass

        # Fallback implementation
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # Calculate corrected p-values
        corrected = np.zeros_like(sorted_p)
        for i in range(n - 1, -1, -1):
            if i == n - 1:
                corrected[i] = sorted_p[i]
            else:
                corrected[i] = min(corrected[i + 1], sorted_p[i] * n / (i + 1))

        # Restore original order
        result = np.zeros_like(p_values)
        result[sorted_indices] = corrected

        return result

    def _fdr_by_correction(self, p_values: np.ndarray) -> np.ndarray:
        """Apply Benjamini-Yekutieli FDR correction."""
        if SCIPY_AVAILABLE:
            try:
                return false_discovery_control(p_values, method='by')
            except BaseException:
                pass

        # Fallback: more conservative than BH
        bh_corrected = self._fdr_bh_correction(p_values)
        n = len(p_values)
        harmonic_sum = np.sum(1.0 / np.arange(1, n + 1))

        return np.minimum(bh_corrected * harmonic_sum, 1.0)


class CrossValidator:
    """Cross-validation for robust statistical analysis."""

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def validate_analysis(self, data: np.ndarray, labels: np.ndarray,
                          analysis_func: callable) -> Dict[str, Any]:
        """Perform cross-validation on analysis function.

        Args:
            data: Input data
            labels: Labels or groups
            analysis_func: Function to validate

        Returns:
            Cross-validation results
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, returning mock results")
            return {"scores": [0.5] * self.n_splits, "mean": 0.5, "std": 0.0}

        try:
            # Create stratified k-fold
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                  random_state=self.random_state)

            scores = []
            for train_idx, test_idx in skf.split(data, labels):
                train_data, test_data = data[train_idx], data[test_idx]
                train_labels, test_labels = labels[train_idx], labels[test_idx]

                # Apply analysis function
                try:
                    score = analysis_func(
                        train_data, test_data, train_labels, test_labels)
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"Analysis function failed in fold: {e}")
                    scores.append(0.0)

            return {
                "scores": scores,
                "mean": np.mean(scores),
                "std": np.std(scores),
                "folds": self.n_splits
            }

        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {"scores": [], "mean": 0.0, "std": 0.0}


class ExperimentLogger:
    """Experiment logging for reproducibility."""

    def __init__(self, experiment_name: str, output_dir: Path):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate experiment ID
        self.experiment_id = f"{experiment_name}_{
            datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize metadata
        self.metadata = ExperimentMetadata(
            experiment_id=self.experiment_id,
            timestamp=datetime.now().isoformat(),
            model_name="",
            model_hash="",
            data_hash="",
            code_hash="",
            hyperparameters={},
            environment_info=self._get_environment_info(),
            random_seeds={},
            results_summary={}
        )

    def log_model(self, model_name: str, model_path: Optional[Path] = None):
        """Log model information."""
        self.metadata.model_name = model_name

        if model_path and model_path.exists():
            self.metadata.model_hash = self._calculate_file_hash(model_path)

    def log_data(self, data_path: Path):
        """Log data information."""
        if data_path.exists():
            self.metadata.data_hash = self._calculate_file_hash(data_path)

    def log_code(self, code_files: List[Path]):
        """Log code information."""
        combined_content = ""
        for file_path in code_files:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    combined_content += f.read()

        self.metadata.code_hash = hashlib.sha256(combined_content.encode()).hexdigest()

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters."""
        self.metadata.hyperparameters.update(hyperparameters)

    def log_random_seeds(self, seeds: Dict[str, int]):
        """Log random seeds for reproducibility."""
        self.metadata.random_seeds.update(seeds)

    def log_results(self, results: Dict[str, Any]):
        """Log experiment results."""
        self.metadata.results_summary.update(results)

    def save_experiment(self, additional_data: Optional[Dict[str, Any]] = None):
        """Save experiment metadata and data."""
        # Save metadata
        metadata_path = self.output_dir / f"{self.experiment_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(self.metadata), f, indent=2)

        # Save additional data if provided
        if additional_data:
            data_path = self.output_dir / f"{self.experiment_id}_data.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump(additional_data, f)

        logger.info(f"Experiment {self.experiment_id} saved to {self.output_dir}")

    def _get_environment_info(self) -> Dict[str, str]:
        """Get environment information."""
        import sys
        import platform

        try:
            import torch
            torch_version = torch.__version__
            cuda_available = str(torch.cuda.is_available())
        except ImportError:
            torch_version = "not installed"
            cuda_available = "unknown"

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch_version,
            "cuda_available": cuda_available
        }

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


def main():
    """Example usage of statistical testing and experiment logging."""
    # Generate sample data
    np.random.seed(42)
    data1 = np.random.normal(0, 1, (100, 768))  # Mock activations
    data2 = np.random.normal(0.2, 1.1, (100, 768))  # Different distribution

    # Statistical testing
    tester = StatisticalTester()

    # Compare activations
    result = tester.compare_activations(data1, data2, test_type="ttest")
    print(f"Statistical test result: {result.interpretation}")

    # Multiple comparison correction
    p_values = [0.01, 0.03, 0.07, 0.12, 0.001]  # Example p-values
    corrector = MultipleComparisonCorrector()

    correction_result = corrector.correct_p_values(p_values, method="fdr_bh")
    print(f"Original p-values: {correction_result.original_p_values}")
    print(
        f"Corrected p-values: {[f'{p:.4f}' for p in correction_result.corrected_p_values]}")
    print(f"Significant tests: {correction_result.significant_tests}")

    # Experiment logging
    logger_obj = ExperimentLogger("test_experiment", Path("./experiment_logs"))
    logger_obj.log_model("test_model")
    logger_obj.log_hyperparameters({"learning_rate": 0.001, "batch_size": 32})
    logger_obj.log_random_seeds({"numpy": 42, "torch": 123})
    logger_obj.log_results({"accuracy": 0.85, "bias_score": 0.12})
    logger_obj.save_experiment()

    print("Scientific rigor demo completed!")


if __name__ == "__main__":
    main()
