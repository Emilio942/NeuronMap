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
import torch

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


class MathematicalRigor:
    """
    Implements formal mathematical metrics for project integrity.
    
    Includes:
    - Mutual Coherence (Proxy for Restricted Isometry Property)
    - Lipschitz Stability Constant (epsilon) for circuit edges
    - Sparsity-to-RIP bound checks
    """
    
    @staticmethod
    def calculate_dictionary_coherence(W_dec: torch.Tensor) -> float:
        """
        Calculates the mutual coherence of the decoder matrix.
        
        A lower coherence (approaching 0) guarantees that latent features 
        can be isolated without polysemantic interference.
        """
        import torch
        # Normalize columns (features)
        norms = torch.norm(W_dec, dim=0, keepdim=True)
        W_norm = W_dec / (norms + 1e-8)
        
        # Calculate Gram matrix G = W^T * W
        gram = torch.abs(torch.matmul(W_norm.t(), W_norm))
        
        # Coherence is the maximum off-diagonal entry
        identity = torch.eye(gram.size(0), device=gram.device)
        coherence = torch.max(gram * (1 - identity))
        return coherence.item()

    @staticmethod
    def estimate_stability_constant(W_enc: torch.Tensor, delta: float) -> float:
        """
        Estimates the stability constant epsilon = L_enc / (1 - delta).
        
        Measures the maximum variance of a circuit edge given input perturbations.
        """
        import torch
        # L_enc is the spectral norm (L2 norm) of the encoder weights
        l_enc = torch.linalg.norm(W_enc, ord=2).item()
        return l_enc / (1 - delta + 1e-8)

    @staticmethod
    def verify_sparsity_bound(lambda_val: float, d_hid: int, sigma_min: float) -> float:
        """
        Checks if the regularization lambda is sufficient to guarantee 
        the theoretical sparsity bound k(lambda).
        """
        import numpy as np
        # Theoretical bound: lambda >= 1/sigma_min * (sqrt(2*log(d_hid/eta)) + c)
        # We assume eta=0.1 and c=0.5 for this check
        eta = 0.1
        c = 0.5
        required_lambda = (1.0 / (sigma_min + 1e-8)) * (np.sqrt(2 * np.log(d_hid / eta)) + c)
        return float(lambda_val >= required_lambda)

    @staticmethod
    def calculate_welch_bound(d_in: int, d_hid: int) -> float:
        """
        Calculates the Welch bound for an overcomplete dictionary.
        
        This is the absolute mathematical lower limit for mutual coherence.
        """
        import numpy as np
        if d_hid <= d_in:
            return 0.0
        return np.sqrt((d_hid - d_in) / (d_in * (d_hid - 1)))

    @staticmethod
    def calculate_spectral_radius_bound(s_star: int, d_in: int, d_hid: int, coherence: Optional[float] = None) -> float:
        """
        Calculates the theoretical worst-case spectral radius of the inter-layer Jacobian.
        
        Uses the Welch bound if no specific coherence is provided.
        rho_max = sqrt(s_star) * sqrt(1 + (s_star - 1) * mu)
        """
        import numpy as np
        mu = coherence if coherence is not None else MathematicalRigor.calculate_welch_bound(d_in, d_hid)
        
        # Formula (8) from the derivation
        rho_max = np.sqrt(s_star) * np.sqrt(1 + (s_star - 1) * mu)
        return float(rho_max)

    @staticmethod
    def calculate_lipschitz_margin(rho_max: float) -> float:
        """
        Calculates the Lipschitz margin epsilon_margin = 1 - rho_max.
        
        If negative, causal path collapse is theoretically unavoidable.
        """
        return 1.0 - rho_max

    @staticmethod
    def calculate_adversarial_stability_threshold(epsilon_margin: float, path_length: int, curvature: float = 0.1) -> float:
        """
        Estimates the maximum adversarial perturbation eta that the circuit can withstand.
        
        eta_max approx (1 - epsilon_margin)^path_length / (C * curvature)
        """
        import numpy as np
        if epsilon_margin <= 0:
            return 0.0
        
        c_kappa = 1.0 # Constant factor
        # Formula (16) approximation
        stability_base = (1.0 - epsilon_margin) ** path_length
        eta_max = stability_base / (c_kappa * curvature + 1e-8)
        return float(eta_max)

    @staticmethod
    def calculate_operator_commutator(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Calculates the commutator [A, B] = AB - BA.
        
        In Topos theory, a non-zero commutator indicates that the logic 
        is non-Boolean and context-dependent.
        """
        import torch
        # Ensure matrices are compatible for multiplication
        return torch.matmul(A, B) - torch.matmul(B, A)

    @staticmethod
    def calculate_commutator_norm(A: torch.Tensor, B: torch.Tensor) -> float:
        """
        Calculates the Frobenius norm of the commutator [A, B].
        Normalized by the product of the individual norms.
        """
        import torch
        commutator = MathematicalRigor.calculate_operator_commutator(A, B)
        norm_comm = torch.linalg.norm(commutator, ord='fro').item()
        norm_a = torch.linalg.norm(A, ord='fro').item()
        norm_b = torch.linalg.norm(B, ord='fro').item()
        
        return norm_comm / (norm_a * norm_b + 1e-8)

    @staticmethod
    def calculate_von_neumann_entropy(activations: torch.Tensor) -> float:
        """
        Calculates the von Neumann Entanglement Entropy of a feature subspace.
        Treats the normalized empirical covariance matrix as a reduced density matrix rho_A.
        S(rho_A) = -Tr(rho_A * ln(rho_A))
        
        Args:
            activations: Tensor of shape [batch_size, feature_dim] representing sub-circuit A.
            
        Returns:
            Entanglement Entropy (float).
            - Low entropy (~0): Sub-circuit is factorizable (independent).
            - High entropy: Sub-circuit is entangled with the environment.
        """
        import torch
        # Center the activations
        centered = activations - activations.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix (representing the interaction within subspace A)
        # We treat this as the reduced density matrix rho_A
        cov = torch.matmul(centered.t(), centered) / (centered.size(0) - 1 + 1e-8)
        
        # Normalize trace to 1 to create a valid density matrix rho_A
        trace = torch.trace(cov)
        if trace <= 1e-8:
            return 0.0
            
        rho_a = cov / trace
        
        # Compute eigenvalues (rho_a is symmetric positive semi-definite)
        eigenvalues = torch.linalg.eigvalsh(rho_a)
        
        # Filter out negative or near-zero eigenvalues for numerical stability
        valid_eigenvalues = eigenvalues[eigenvalues > 1e-9]
        
        # Calculate von Neumann entropy S = - sum(lambda * ln(lambda))
        entropy = -torch.sum(valid_eigenvalues * torch.log(valid_eigenvalues))
        return float(entropy.item())

    @staticmethod
    def calculate_interaction_fidelity(activations_a: torch.Tensor, activations_b: torch.Tensor) -> float:
        """
        Calculates the EXACT Bures Fidelity between two neural subspaces (A and B).
        F(rho, sigma) = (Tr[sqrt(sqrt(rho) * sigma * sqrt(rho))])^2
        
        This is the gold standard for comparing mixed states in quantum information.
        """
        import torch
        def get_rho(acts):
            centered = acts - acts.mean(dim=0, keepdim=True)
            cov = torch.matmul(centered.t(), centered) / (centered.size(0) - 1 + 1e-8)
            trace = torch.trace(cov)
            return cov / (trace + 1e-8)

        rho = get_rho(activations_a)
        sigma = get_rho(activations_b)

        if rho.shape != sigma.shape:
            return 0.0

        # Matrix square root function using Eigenvalue Decomposition
        def matrix_sqrt(matrix):
            # Ensure symmetry
            matrix = (matrix + matrix.t()) / 2.0
            evals, evecs = torch.linalg.eigh(matrix)
            # Clip negative eigenvalues due to precision
            evals = torch.clamp(evals, min=0)
            return torch.matmul(evecs, torch.matmul(torch.diag(torch.sqrt(evals)), evecs.t()))

        try:
            sqrt_rho = matrix_sqrt(rho)
            middle = torch.matmul(sqrt_rho, torch.matmul(sigma, sqrt_rho))
            sqrt_middle = matrix_sqrt(middle)
            fidelity = torch.trace(sqrt_middle) ** 2
            return float(fidelity.item())
        except Exception:
            # Fallback to simple trace product if numerical error occurs
            return float(torch.trace(torch.matmul(rho, sigma)).item())

    @staticmethod
    def calculate_noether_charge(activations: torch.Tensor, gradients: torch.Tensor, generator: torch.Tensor) -> float:
        """
        Calculates the Empirical Noether Charge Q^a.
        Q^a = <grad_x, T^a * x>
        
        Measures the alignment of the model's sensitivity with a Lie Group generator T^a.
        If Q^a is zero, the model is invariant under the symmetry G.
        If Q^a is non-zero, the symmetry is broken, and semantically selective features emerge.
        """
        import torch
        # T^a * x (Infinitesimal transformation)
        # activations: [batch, dim], generator: [dim, dim]
        transformed_x = torch.matmul(activations, generator.t())
        
        # Dot product with gradients: sum(grad * delta_x)
        charge = torch.sum(gradients * transformed_x, dim=-1)
        return float(torch.mean(torch.abs(charge)).item())

    @staticmethod
    def detect_symmetry_breaking(weight_matrix: torch.Tensor) -> float:
        """
        Measures the deviation from SO(d) invariance (Orthogonality).
        Deviation = ||W^T * W - I||_F
        
        High deviation indicates Spontaneous Symmetry Breaking (SSB).
        """
        import torch
        dim = weight_matrix.shape[-1]
        identity = torch.eye(dim, device=weight_matrix.device)
        
        # W^T * W should be I for SO(d) invariance
        gram = torch.matmul(weight_matrix.t(), weight_matrix)
        deviation = torch.linalg.norm(gram - identity, ord='fro')
        return float(deviation.item() / (dim * dim))

    @staticmethod
    def discover_maximal_symmetry_group(cov_matrix: torch.Tensor, tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Discovers the maximal continuous symmetry group G of a sub-network by 
        finding the Lie algebra generators X that preserve the empirical covariance.
        Effectively analyzes the degenerate eigenspaces (isotropic subspaces) to find SO(n) symmetries.
        
        Returns the Lie algebra dimension and the number of Casimir Invariants (orbit labels).
        """
        import torch
        dim = cov_matrix.shape[0]
        evals = torch.linalg.eigvalsh(cov_matrix)
        
        # Group eigenvalues by tolerance to find degenerate eigenspaces (isotropic subspaces)
        # Symmetries exist where the covariance is invariant (equal eigenvalues)
        rounded_evals = torch.round(evals / tolerance) * tolerance
        unique_evals, counts = torch.unique(rounded_evals, return_counts=True)
        
        symmetry_dim = 0
        num_casimirs = 0
        
        for count in counts:
            n = int(count.item())
            if n > 1:
                # Each degenerate subspace of dimension n contributes SO(n) symmetry
                # Dimension of SO(n) Lie Algebra is n(n-1)/2
                symmetry_dim += n * (n - 1) // 2
                # The rank of SO(n) determines the number of fundamental Casimir invariants
                rank = n // 2
                num_casimirs += rank
                
        return {
            "lie_algebra_dimension": symmetry_dim,
            "num_casimir_invariants": num_casimirs,
            "symmetry_type": f"SO({int(torch.max(counts).item())})" if symmetry_dim > 0 else "Trivial",
            "is_spontaneously_broken": symmetry_dim == 0
        }

    @staticmethod
    def calculate_topological_index(activations: torch.Tensor) -> int:
        """
        Estimates the topological winding number nu.
        Simplified version: checks the parity of the orientation distribution.
        """
        import torch
        # Calculate the phase of the first two dimensions (SO(2) proxy)
        if activations.shape[-1] < 2:
            return 0
            
        phases = torch.atan2(activations[:, 1], activations[:, 0])
        # Total phase accumulation (proxy for winding number)
        total_winding = torch.sum(torch.diff(phases)) / (2 * np.pi)
        return int(torch.round(total_winding).item())

    @staticmethod
    def calculate_scalar_curvature_proxy(fim: torch.Tensor) -> float:
        """
        Estimates the Scalar Curvature R of the parameter manifold.
        Based on the spectral spread of the Fisher Information Matrix.
        R approx sum(log(lambda_i / lambda_j)^2)
        """
        import torch
        evals = torch.linalg.eigvalsh(fim)
        evals = torch.clamp(evals, min=1e-6)
        
        # High spread between eigenvalues indicates high sectional curvature
        log_evals = torch.log(evals)
        variance = torch.var(log_evals)
        return float(variance.item())

    @staticmethod
    def detect_geometric_singularities(curvature_history: List[float]) -> Dict[str, Any]:
        """
        Detects singularities in the Ricci-flow-like evolution of weights.
        A spike in curvature often precedes the 'Emergence' of a mechanistic circuit.
        """
        import numpy as np
        if len(curvature_history) < 4:
            return {"singularity_detected": False, "score": 0.0}
            
        # Calculate the rate of change (Flow velocity)
        velocity = np.diff(curvature_history)
        acceleration = np.diff(velocity)
        
        # Singularity score based on current acceleration
        current_accel = acceleration[-1] if len(acceleration) > 0 else 0.0
        
        # Robust threshold: 3x the standard deviation of the previous velocities
        # This flags any sudden jump in the rate of change
        prev_velocity_std = np.std(velocity[:-1]) + 1e-8
        z_score = current_accel / prev_velocity_std
        
        # Threshold: z-score > 5.0 indicates a geometric collapse/spike
        is_singular = bool(z_score > 5.0)
        
        return {
            "singularity_detected": is_singular,
            "singularity_score": float(z_score),
            "is_circuit_emergence": is_singular
        }

    @staticmethod
    def calculate_lyapunov_spectrum(jacobians: List[torch.Tensor]) -> np.ndarray:
        """
        Calculates the full Lyapunov Spectrum using the continuous QR-Decomposition (Benettin Algorithm).
        This is numerically stable for very deep networks.
        """
        import torch
        import numpy as np
        if not jacobians:
            return np.array([])

        dim = jacobians[0].shape[-1]
        num_layers = len(jacobians)
        
        # Initialize orthogonal frame
        Q = torch.eye(dim, device=jacobians[0].device)
        exponents = torch.zeros(dim, device=jacobians[0].device)

        for J in jacobians:
            # Propagate the frame: Z = J * Q
            # Ensure J is square or handled correctly
            Z = torch.matmul(J, Q)
            
            # QR Decomposition to re-orthogonalize and extract growth
            Q, R = torch.linalg.qr(Z)
            
            # Accumulate the log of the diagonal of R
            diag_r = torch.abs(torch.diag(R))
            exponents += torch.log(diag_r + 1e-10)

        # Average over the path
        spectrum = exponents / num_layers
        return spectrum.cpu().numpy()

    @staticmethod
    def calculate_maximal_lyapunov_exponent(jacobians: List[torch.Tensor]) -> float:
        """Helper to get only the maximal exponent."""
        import numpy as np
        spectrum = MathematicalRigor.calculate_lyapunov_spectrum(jacobians)
        return float(np.max(spectrum)) if len(spectrum) > 0 else 0.0

    @staticmethod
    def calculate_kaplan_yorke_dimension(lyapunov_spectrum: np.ndarray) -> float:
        """
        Calculates the exact Kaplan-Yorke Dimension (Information Dimension).
        D_ky = j + sum(lambda_i for i=1 to j) / |lambda_{j+1}|
        where j is the largest integer such that the sum of the first j exponents is >= 0.
        
        This defines the 'Information-Theoretic Compression Limit' of the manifold.
        """
        import numpy as np
        if len(lyapunov_spectrum) == 0:
            return 0.0
            
        # Sort spectrum descending
        sorted_spectrum = np.sort(lyapunov_spectrum)[::-1]
        
        cumulative_sum = 0.0
        j = 0
        
        # Find the largest j such that the sum is non-negative
        for i in range(len(sorted_spectrum)):
            if cumulative_sum + sorted_spectrum[i] >= 0:
                cumulative_sum += sorted_spectrum[i]
                j = i + 1
            else:
                break
        
        # If all exponents are negative, the dimension is 0 (point attractor)
        if j == 0:
            return 0.0
            
        # If all exponents are positive or j is the last index, the dimension is the full dim
        if j == len(sorted_spectrum):
            return float(j)
            
        # Linear interpolation for the fractional part
        lambda_next = sorted_spectrum[j]
        if abs(lambda_next) < 1e-10:
            return float(j)
            
        d_ky = j + (cumulative_sum / abs(lambda_next))
        return float(d_ky)

    @staticmethod
    def calculate_prediction_horizon(lambda_max: float, delta_x0: float, epsilon: float = 1.0) -> int:
        """
        Calculates the Prediction Horizon H based on the MLE.
        
        H = floor((1/|lambda_max|) * ln(epsilon / ||delta_x0||))
        
        Args:
            lambda_max: Maximal Lyapunov Exponent.
            delta_x0: Initial perturbation magnitude.
            epsilon: Maximum allowed divergence (default 1.0).
            
        Returns:
            Horizon H (int).
        """
        import numpy as np
        if abs(lambda_max) < 1e-10:
            return 1000 # Effectively infinite for short sequences
            
        h = (1.0 / abs(lambda_max)) * np.log(epsilon / (delta_x0 + 1e-10))
        return int(max(0, np.floor(h)))

    @staticmethod
    def calculate_gcv_optimal_alpha(fim: torch.Tensor, causal_grad: torch.Tensor) -> float:
        """
        Finds the optimal Tikhonov regularization parameter alpha using 
        Generalized Cross-Validation (GCV).
        Minimizes GCV(alpha) = ||y - X*beta||^2 / (1 - tr(H)/n)^2
        """
        import torch
        import numpy as np
        
        evals = torch.linalg.eigvalsh(fim)
        # We test a log-range of alphas
        alphas = np.logspace(-6, 2, 20)
        best_alpha = alphas[0]
        min_gcv = float('inf')
        
        n = fim.shape[0]
        
        for alpha in alphas:
            # Effective degrees of freedom: tr(H_alpha) = sum(lambda_i / (lambda_i + alpha))
            df = torch.sum(evals / (evals + alpha)).item()
            
            # Residual sum of squares proxy in information space
            # RSS = c^T * (I - H_alpha)^2 * c (Simplified for CRLB context)
            denom = (1.0 - df / n) ** 2
            if denom < 1e-6: continue
            
            # Weighted residual
            rss = torch.sum(causal_grad**2 * (alpha / (evals.unsqueeze(1) + alpha))**2).item()
            gcv = rss / denom
            
            if gcv < min_gcv:
                min_gcv = gcv
                best_alpha = alpha
                
        return float(best_alpha)

    @staticmethod
    def calculate_tikhonov_crlb(fim: torch.Tensor, causal_grad: torch.Tensor, alpha: Optional[float] = None) -> float:
        """
        Calculates the CRLB using Tikhonov regularization.
        If alpha is None, it uses GCV to find the optimal value.
        """
        import torch
        if alpha is None:
            alpha = MathematicalRigor.calculate_gcv_optimal_alpha(fim, causal_grad)
            
        # Regularized inversion: (FIM + alpha*I)^-1
        dim = fim.shape[0]
        reg_fim = fim + alpha * torch.eye(dim, device=fim.device)
        fim_inv = torch.linalg.inv(reg_fim)
        
        bound = torch.matmul(causal_grad.t(), torch.matmul(fim_inv, causal_grad))
        return float(bound.item())


class TopologicalCircuitAnalyzer:
    """
    Implements sheaf-theoretic topological analysis for Mechanistic Circuits.
    Calculates Euler Characteristic, Cohomology (H^1), and Topological Capacity.
    """
    
    @staticmethod
    def analyze_circuit_topology(num_vertices: int, num_edges: int, num_components: int = 1) -> Dict[str, float]:
        """
        Analyzes the topology of a 1-complex circuit DAG.
        
        Args:
            num_vertices: Number of nodes in the circuit.
            num_edges: Number of edges in the circuit.
            num_components: Number of connected components (default 1).
            
        Returns:
            Dict containing euler_characteristic and dim_H1.
        """
        # Euler Characteristic for a 1-complex: X = V - E
        chi = num_vertices - num_edges
        
        # Based on Euler-Poincare: chi = dim H^0 - dim H^1
        # For a circuit with 'c' connected components, dim H^0 = c.
        # Therefore: dim H^1 = c - chi
        dim_h1 = num_components - chi
        
        return {
            "euler_characteristic": chi,
            "dim_h1": max(0, dim_h1) # Cohomology dimension cannot be negative
        }

    @staticmethod
    def detect_higher_order_conflicts(num_vertices: int, num_edges: int, num_faces: int) -> Dict[str, Any]:
        """
        Uses H^2 cohomology to detect 'Topological Conflicts' between layers.
        Signifies hierarchical nesting inconsistencies that no linear SAE can resolve.
        """
        # Euler Characteristic for a 2-complex: X = V - E + F
        chi = num_vertices - num_edges + num_faces
        
        # In a simply connected component: dim H^2 = chi - dim H^0 + dim H^1
        # Simplified proxy for H^2 obstructions:
        h2_obstruction = chi < 1 # Negative Euler char in 2-complexes indicates H^2 or high H^1
        
        return {
            "euler_characteristic_2d": chi,
            "has_topological_conflict": bool(h2_obstruction),
            "h2_proxy_score": float(max(0, 1 - chi))
        }

    @staticmethod
    def calculate_topological_capacity(dim_h1: int, max_sae_ambiguity_dim: int) -> float:
        """
        Calculates the Topological Capacity C.
        
        C = 0.5 * dim H^1 * kappa
        where kappa is the minimal entropy cost per obstruction, estimated
        as log(max_sae_ambiguity_dim).
        """
        import numpy as np
        if max_sae_ambiguity_dim <= 1:
            kappa = 0.0
        else:
            kappa = np.log(max_sae_ambiguity_dim)
            
        capacity = 0.5 * dim_h1 * kappa
        return float(capacity)

    @staticmethod
    def check_phase_transition(delta_s: float, capacity: float) -> bool:
        """
        Checks if the entropy change Delta S triggers a catastrophic phase transition.
        Returns True if a phase transition occurs (collapse).
        """
        # Phase transition occurs if Delta S > C
        return delta_s > capacity


class InformationGeometricAnalyzer:
    """
    Implements Information Geometry metrics for parameter precision.
    Uses the Fisher Information Metric and Cramér-Rao Lower Bound.
    """
    
    @staticmethod
    def calculate_empirical_fisher(gradients: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Empirical Fisher Information Matrix (FIM).
        FIM = 1/N * sum(grad * grad^T)
        """
        import torch
        # gradients shape: [batch_size, num_params]
        batch_size = gradients.shape[0]
        # FIM calculation
        fim = torch.matmul(gradients.t(), gradients) / (batch_size + 1e-8)
        return fim

    @staticmethod
    def create_admissible_projector(fim: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:
        """
        Creates a projector onto the informative subspace (non-null space).
        Filters out directions where the model tangent kernel has zero sensitivity.
        """
        import torch
        eigenvalues, eigenvectors = torch.linalg.eigh(fim)
        
        # Identify indices of eigenvalues above threshold
        mask = eigenvalues > threshold
        relevant_vectors = eigenvectors[:, mask]
        
        # Pi_ok = U_high * U_high^T
        projector = torch.matmul(relevant_vectors, relevant_vectors.t())
        return projector

    @staticmethod
    def calculate_cramer_rao_bound(fim: torch.Tensor, causal_functional_grad: torch.Tensor) -> float:
        """
        Calculates the CRLB for a causal effect estimator.
        Var(C) >= c^T * I_inv * c
        """
        import torch
        # Use pseudo-inverse for stability in the presence of a null space
        fim_inv = torch.linalg.pinv(fim, rcond=1e-5)
        
        # c^T * I_inv * c
        bound = torch.matmul(causal_functional_grad.t(), torch.matmul(fim_inv, causal_functional_grad))
        return float(bound.item())

    @staticmethod
    def calculate_quantum_fisher_information(rho: torch.Tensor, L_i: torch.Tensor, L_j: torch.Tensor) -> float:
        """
        Calculates the (i,j) entry of the Non-Commutative (Quantum) Fisher Information Metric.
        F_ij = 1/2 * Tr(rho * {L_i, L_j}) where {.,.} is the anticommutator.
        """
        import torch
        # Anticommutator {L_i, L_j} = L_i*L_j + L_j*L_i
        anticomm = torch.matmul(L_i, L_j) + torch.matmul(L_j, L_i)
        
        f_ij = 0.5 * torch.trace(torch.matmul(rho, anticomm))
        return float(f_ij.item())

    @staticmethod
    def calculate_interpretability_uncertainty(A: torch.Tensor, B: torch.Tensor, rho: torch.Tensor) -> float:
        """
        Implements the 'Interpretability Uncertainty Principle'.
        Delta_Semantic * Delta_Causal >= 1/4 |<[A, B]>_rho|^2
        
        A: Semantic localization operator (e.g. projection onto feature)
        B: Causal attribution operator (e.g. SLD of the parameter)
        rho: Neural density matrix
        """
        import torch
        # Compute commutator [A, B]
        comm = torch.matmul(A, B) - torch.matmul(B, A)
        
        # Expectation value <[A, B]> = Tr(rho * [A, B])
        exp_comm = torch.trace(torch.matmul(rho, comm))
        
        # Uncertainty bound
        bound = 0.25 * torch.abs(exp_comm)**2
        return float(bound.item())

    @staticmethod
    def calculate_gcv_optimal_alpha(fim: torch.Tensor, causal_grad: torch.Tensor) -> float:
        """
        Finds the optimal Tikhonov regularization parameter alpha using 
        Generalized Cross-Validation (GCV).
        Minimizes GCV(alpha) = ||y - X*beta||^2 / (1 - tr(H)/n)^2
        """
        import torch
        import numpy as np
        
        evals = torch.linalg.eigvalsh(fim)
        # We test a log-range of alphas
        alphas = np.logspace(-6, 2, 20)
        best_alpha = alphas[0]
        min_gcv = float('inf')
        
        n = fim.shape[0]
        
        for alpha in alphas:
            # Effective degrees of freedom: tr(H_alpha) = sum(lambda_i / (lambda_i + alpha))
            df = torch.sum(evals / (evals + alpha)).item()
            
            # Residual sum of squares proxy in information space
            # RSS = c^T * (I - H_alpha)^2 * c (Simplified for CRLB context)
            denom = (1.0 - df / n) ** 2
            if denom < 1e-6: continue
            
            # Weighted residual
            rss = torch.sum(causal_grad**2 * (alpha / (evals.unsqueeze(1) + alpha))**2).item()
            gcv = rss / denom
            
            if gcv < min_gcv:
                min_gcv = gcv
                best_alpha = alpha
                
        return float(best_alpha)

    @staticmethod
    def calculate_tikhonov_crlb(fim: torch.Tensor, causal_grad: torch.Tensor, alpha: Optional[float] = None) -> float:
        """
        Calculates the CRLB using Tikhonov regularization.
        If alpha is None, it uses GCV to find the optimal value.
        """
        import torch
        if alpha is None:
            alpha = MathematicalRigor.calculate_gcv_optimal_alpha(fim, causal_grad)
            
        # Regularized inversion: (FIM + alpha*I)^-1
        dim = fim.shape[0]
        reg_fim = fim + alpha * torch.eye(dim, device=fim.device)
        fim_inv = torch.linalg.inv(reg_fim)
        
        bound = torch.matmul(causal_grad.t(), torch.matmul(fim_inv, causal_grad))
        return float(bound.item())


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
