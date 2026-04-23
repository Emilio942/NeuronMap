import torch
import numpy as np
import pytest
from src.analysis.scientific_rigor import MathematicalRigor

def test_mathematical_rigor_coherence():
    print("\nTesting Mathematical Rigor: Mutual Coherence")
    
    # 1. Test case: Orthogonal dictionary (ideal case)
    # 3 features, 3 dimensions
    W_ortho = torch.eye(3)
    coherence_ortho = MathematicalRigor.calculate_dictionary_coherence(W_ortho)
    print(f"Ortho coherence: {coherence_ortho}")
    assert coherence_ortho < 1e-6 # Should be nearly 0
    
    # 2. Test case: Highly correlated dictionary (bad case)
    # Two features are nearly identical in a 2D space
    W_bad = torch.tensor([
        [1.0, 0.999],
        [0.0, 0.001]
    ]) # 2 dims, 2 features (columns)
    coherence_bad = MathematicalRigor.calculate_dictionary_coherence(W_bad)
    print(f"Bad coherence: {coherence_bad}")
    assert coherence_bad > 0.9 # Should be close to 1
    
    print("✓ Mutual Coherence metric validated")

def test_mathematical_rigor_stability():
    print("\nTesting Mathematical Rigor: Stability Constant (epsilon)")
    
    # Mock encoder weights
    W_enc = torch.randn(128, 64) * 0.1
    
    # Case A: Low delta (stable)
    epsilon_stable = MathematicalRigor.estimate_stability_constant(W_enc, delta=0.1)
    
    # Case B: High delta (unstable/superposition)
    epsilon_unstable = MathematicalRigor.estimate_stability_constant(W_enc, delta=0.8)
    
    print(f"Stable epsilon: {epsilon_stable}")
    print(f"Unstable epsilon: {epsilon_unstable}")
    
    assert epsilon_unstable > epsilon_stable
    print("✓ Stability Constant metric validated")

def test_sparsity_bound():
    print("\nTesting Mathematical Rigor: Sparsity Bound")
    # Small hidden dim, high sigma_min (stable dictionary)
    is_sufficient = MathematicalRigor.verify_sparsity_bound(lambda_val=5.0, d_hid=100, sigma_min=0.9)
    print(f"Lambda=5.0 sufficient? {is_sufficient}")
    assert is_sufficient == 1.0
    
    # Very high hidden dim, low sigma_min (unstable)
    is_insufficient = MathematicalRigor.verify_sparsity_bound(lambda_val=0.1, d_hid=10000, sigma_min=0.1)
    print(f"Lambda=0.1 sufficient? {is_insufficient}")
    assert is_insufficient == 0.0
    print("✓ Sparsity Bound verification validated")

def test_advanced_spectral_rigor():
    print("\nTesting Advanced Mathematical Rigor: Spectral Analysis")
    
    d_in = 64
    d_hid = 256 # Overcomplete
    s_star = 5   # Target sparsity
    
    # 1. Welch Bound
    mu_welch = MathematicalRigor.calculate_welch_bound(d_in, d_hid)
    print(f"Welch Bound for {d_in}->{d_hid}: {mu_welch}")
    assert 0 < mu_welch < 1
    
    # 2. Spectral Radius
    # Using ideal Welch bound coherence
    rho_max_welch = MathematicalRigor.calculate_spectral_radius_bound(s_star, d_in, d_hid)
    # Using real (higher) coherence
    rho_max_real = MathematicalRigor.calculate_spectral_radius_bound(s_star, d_in, d_hid, coherence=0.2)
    
    print(f"Rho_max (Welch): {rho_max_welch}")
    print(f"Rho_max (Real): {rho_max_real}")
    
    assert rho_max_real > rho_max_welch
    
    # 3. Lipschitz Margin
    # If rho_max is < 1, margin should be positive
    margin = MathematicalRigor.calculate_lipschitz_margin(rho_max=0.8)
    print(f"Margin for rho=0.8: {margin}")
    assert margin == pytest.approx(0.2)
    
    # 4. Adversarial Stability
    eta_max = MathematicalRigor.calculate_adversarial_stability_threshold(epsilon_margin=0.2, path_length=3)
    print(f"Adversarial stability threshold eta_max: {eta_max}")
    assert eta_max > 0
    
    print("✓ Advanced Spectral Analysis validated")

def test_topological_circuit_analyzer():
    from src.analysis.scientific_rigor import TopologicalCircuitAnalyzer
    print("\nTesting Topological Circuit Analyzer: Sheaf Cohomology")
    
    # Test Case 1: Simple Tree (No cycles, H^1 = 0)
    # 5 vertices, 4 edges
    tree_topology = TopologicalCircuitAnalyzer.analyze_circuit_topology(num_vertices=5, num_edges=4, num_components=1)
    print(f"Tree Topology: {tree_topology}")
    assert tree_topology["euler_characteristic"] == 1
    assert tree_topology["dim_h1"] == 0
    
    # Test Case 2: Graph with 2 cycles (H^1 = 2)
    # 5 vertices, 6 edges
    cycle_topology = TopologicalCircuitAnalyzer.analyze_circuit_topology(num_vertices=5, num_edges=6, num_components=1)
    print(f"Cycle Topology: {cycle_topology}")
    assert cycle_topology["euler_characteristic"] == -1
    assert cycle_topology["dim_h1"] == 2
    
    # Test Case 3: Topological Capacity
    # dim_h1 = 2, max_sae_ambiguity = 10 -> kappa = ln(10) ~ 2.3
    # Capacity = 0.5 * 2 * ln(10) = ln(10)
    capacity = TopologicalCircuitAnalyzer.calculate_topological_capacity(dim_h1=2, max_sae_ambiguity_dim=10)
    print(f"Topological Capacity: {capacity}")
    assert capacity == pytest.approx(np.log(10))
    
    # Test Case 4: Phase Transition
    delta_s_safe = 2.0 # Less than ln(10) ~ 2.3
    delta_s_collapse = 3.0 # Greater than ln(10)
    
    assert not TopologicalCircuitAnalyzer.check_phase_transition(delta_s_safe, capacity)
    assert TopologicalCircuitAnalyzer.check_phase_transition(delta_s_collapse, capacity)
    
    # Test Case 5: Higher-Order Conflicts (H^2)
    # Sphere-like topology: Tetrahedron (V=4, E=6, F=4)
    # Boundary d1: 4x6 matrix
    d1_sphere = torch.tensor([
        [-1, -1, -1,  0,  0,  0],
        [ 1,  0,  0, -1, -1,  0],
        [ 0,  1,  0,  1,  0, -1],
        [ 0,  0,  1,  0,  1,  1]
    ], dtype=torch.float32)
    # Boundary d2: 6x4 matrix (one face per triplet)
    # This is a simplification for testing the rank logic
    d2_sphere = torch.tensor([
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, -1, 0],
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1]
    ], dtype=torch.float32)
    
    sphere_conflict = TopologicalCircuitAnalyzer.detect_higher_order_conflicts(num_vertices=4, boundary_1=d1_sphere, boundary_2=d2_sphere)
    print(f"Sphere (Tetrahedron) Betti: {sphere_conflict['betti_numbers']}")
    # beta_0 = 4 - 3 = 1 (one component)
    # beta_2 = 4 - 3 = 1 (one cavity)
    assert sphere_conflict["betti_numbers"]["beta_0"] == 1
    assert sphere_conflict["betti_numbers"]["beta_2"] == 1
    assert sphere_conflict["has_topological_conflict"]
    
    print("✓ Topological Circuit Analyzer validated")

def test_operator_commutator():
    print("\nTesting Mathematical Rigor: Operator Commutator")
    
    # 1. Test Case: Commuting matrices (Diagonal matrices always commute)
    A = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
    B = torch.diag(torch.tensor([4.0, 5.0, 6.0]))
    norm_comm_zero = MathematicalRigor.calculate_commutator_norm(A, B)
    print(f"Commuting norm: {norm_comm_zero}")
    assert norm_comm_zero < 1e-6
    
    # 2. Test Case: Non-commuting matrices
    # Canonical non-commuting: [[1, 1], [0, 1]] and [[1, 0], [1, 1]]
    A_nc = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    B_nc = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    # AB = [[2, 1], [1, 1]], BA = [[1, 1], [1, 2]]
    # [A, B] = [[1, 0], [0, -1]] -> Norm is sqrt(2) ~ 1.414
    norm_comm_nc = MathematicalRigor.calculate_commutator_norm(A_nc, B_nc)
    print(f"Non-commuting norm: {norm_comm_nc}")
    assert norm_comm_nc > 0.1
    
    print("✓ Operator Commutator logic validated")

def test_entanglement_entropy():
    from src.analysis.scientific_rigor import MathematicalRigor
    print("\nTesting Mathematical Rigor: von Neumann Entanglement Entropy")
    
    # 1. Pure State (Factorizable)
    # Perfectly correlated activations (rank 1)
    # 100 samples, 10 dims
    base = torch.randn(100, 1)
    activations_pure = base * torch.ones(1, 10) # All dims identical
    
    entropy_pure = MathematicalRigor.calculate_von_neumann_entropy(activations_pure)
    print(f"Pure state entropy: {entropy_pure}")
    # Ideal S(rho) = 0 for rank 1
    assert entropy_pure < 1e-4
    
    # 2. Mixed/Entangled State (Noise)
    # High-dimensional random noise (rank 10)
    activations_mixed = torch.randn(100, 10)
    entropy_mixed = MathematicalRigor.calculate_von_neumann_entropy(activations_mixed)
    print(f"Mixed state entropy: {entropy_mixed}")
    
    # Random noise should have high entropy (S ~ ln(dim))
    assert entropy_mixed > 1.0
    assert entropy_mixed > entropy_pure
    
    # 3. Fidelity (EXACT Bures Fidelity)
    fidelity_self = MathematicalRigor.calculate_interaction_fidelity(activations_mixed, activations_mixed)
    print(f"Self-Fidelity: {fidelity_self}")
    # Fidelity between identical states should be 1.0
    assert fidelity_self == pytest.approx(1.0, abs=1e-2)
    
    # Orthogonal subspaces: [1, 0, 0...] vs [0, 1, 0...]
    acts_a = torch.zeros(100, 10)
    acts_a[:, 0] = torch.randn(100)
    acts_b = torch.zeros(100, 10)
    acts_b[:, 1] = torch.randn(100)
    
    fidelity_other = MathematicalRigor.calculate_interaction_fidelity(acts_a, acts_b)
    print(f"Other-Fidelity (Orthogonal): {fidelity_other}")
    assert fidelity_other < 0.1
    
    print("✓ von Neumann Entanglement Entropy & Bures Fidelity validated")

def test_lyapunov_stability():
    from src.analysis.scientific_rigor import MathematicalRigor
    print("\nTesting Mathematical Rigor: Lyapunov Stability (QR-Benettin)")
    
    # 1. Stable dynamics (Jacobian < I)
    dim = 4
    jacobians_stable = [0.9 * torch.eye(dim) for _ in range(10)]
    spectrum_stable = MathematicalRigor.calculate_lyapunov_spectrum(jacobians_stable)
    print(f"Stable Lyapunov Spectrum: {spectrum_stable}")
    # All exponents should be ln(0.9) ~ -0.105
    assert np.all(spectrum_stable < 0)
    assert len(spectrum_stable) == dim
    
    # 2. Mixed dynamics
    # One expanding, others contracting
    J_mixed = torch.eye(dim)
    J_mixed[0, 0] = 1.2 # Expanding
    J_mixed[1, 1] = 0.5 # Strongly contracting
    
    jacobians_mixed = [J_mixed for _ in range(10)]
    spectrum_mixed = MathematicalRigor.calculate_lyapunov_spectrum(jacobians_mixed)
    print(f"Mixed Lyapunov Spectrum: {spectrum_mixed}")
    assert np.max(spectrum_mixed) > 0 # At least one positive
    assert np.min(spectrum_mixed) < 0 # At least one negative
    
    # 3. Prediction Horizon
    # lambda = -0.1, delta_x0 = 1e-3, epsilon = 1.0
    # H = floor(1/0.1 * ln(1/1e-3)) = floor(10 * 6.9) = 69
    horizon = MathematicalRigor.calculate_prediction_horizon(lambda_max=-0.1, delta_x0=1e-3)
    print(f"Prediction Horizon: {horizon}")
    assert horizon >= 60
    
    # 4. Kaplan-Yorke Dimension
    # Spectrum: [0.5, 0.2, -0.4, -0.8]
    # sum(0.5) = 0.5 (>=0, j=1)
    # sum(0.5, 0.2) = 0.7 (>=0, j=2)
    # sum(0.7, -0.4) = 0.3 (>=0, j=3)
    # sum(0.3, -0.8) = -0.5 (<0) -> j=3
    # D_ky = 3 + (0.3 / |-0.8|) = 3 + 0.375 = 3.375
    spectrum = np.array([0.5, 0.2, -0.4, -0.8])
    d_ky = MathematicalRigor.calculate_kaplan_yorke_dimension(spectrum)
    print(f"Kaplan-Yorke Dimension: {d_ky}")
    assert d_ky == pytest.approx(3.375)
    
    print("✓ Lyapunov Stability analysis validated")

def test_symmetry_noether():
    print("\nTesting Mathematical Rigor: Noether Symmetries & SSB")
    
    # 1. Invariant Case (Zero Charge)
    dim = 4
    activations = torch.randn(10, dim)
    # Generator for SO(2) rotation in first two dims: [[0, 1], [-1, 0]]
    generator = torch.zeros(dim, dim)
    generator[0, 1] = 1.0
    generator[1, 0] = -1.0
    
    # Gradient that is perfectly orthogonal to the rotation (invariant)
    # Gradient is parallel to the radius vector
    gradients_inv = activations.clone()
    charge_inv = MathematicalRigor.calculate_noether_charge(activations, gradients_inv, generator)
    print(f"Invariant Noether Charge: {charge_inv}")
    assert charge_inv < 1e-5
    
    # 2. Broken Symmetry (Non-zero Charge)
    # Gradient that follows the rotation
    gradients_broken = torch.matmul(activations, generator.t())
    charge_broken = MathematicalRigor.calculate_noether_charge(activations, gradients_broken, generator)
    print(f"Broken Noether Charge: {charge_broken}")
    assert charge_broken > 0.5
    
    # 3. Symmetry Breaking Detection (SSB)
    # Case A: Orthogonal Matrix (No SSB)
    W_ortho, _ = torch.linalg.qr(torch.randn(dim, dim))
    ssb_low = MathematicalRigor.detect_symmetry_breaking(W_ortho)
    print(f"SSB Deviation (Ortho): {ssb_low}")
    assert ssb_low < 1e-5
    
    # Case B: Random Matrix (High SSB)
    W_random = torch.randn(dim, dim)
    ssb_high = MathematicalRigor.detect_symmetry_breaking(W_random)
    print(f"SSB Deviation (Random): {ssb_high}")
    assert ssb_high > ssb_low
    
    # 4. Maximal Symmetry Discovery
    # Perfectly isotropic covariance (e.g., SO(4) symmetry)
    cov_symmetric = torch.eye(4) * 2.5
    sym_result = MathematicalRigor.discover_maximal_symmetry_group(cov_symmetric)
    print(f"Symmetry Discovery (Isotropic): {sym_result}")
    assert sym_result["lie_algebra_dimension"] == 6 # 4*3/2 for SO(4)
    assert sym_result["num_casimir_invariants"] == 2 # rank of SO(4) is 2
    assert not sym_result["is_spontaneously_broken"]
    
    # Broken Symmetry (Anisotropic covariance, distinct eigenvalues)
    cov_broken = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    broken_sym_result = MathematicalRigor.discover_maximal_symmetry_group(cov_broken)
    print(f"Symmetry Discovery (Broken): {broken_sym_result}")
    assert broken_sym_result["lie_algebra_dimension"] == 0
    assert broken_sym_result["is_spontaneously_broken"]
    
    print("✓ Noether Symmetry & SSB detection validated")

def test_information_geometry():
    from src.analysis.scientific_rigor import InformationGeometricAnalyzer
    print("\nTesting Mathematical Rigor: Information Geometry (Fisher & CRLB)")
    
    # 1. Empirical Fisher
    # 100 samples, 10 parameters
    grads = torch.randn(100, 10)
    fim = InformationGeometricAnalyzer.calculate_empirical_fisher(grads)
    print(f"FIM shape: {fim.shape}")
    assert fim.shape == (10, 10)
    # FIM should be positive semi-definite (all eigenvalues >= 0)
    eigenvalues = torch.linalg.eigvalsh(fim)
    assert torch.all(eigenvalues > -1e-5)
    
    # 2. Admissible Projector
    projector = InformationGeometricAnalyzer.create_admissible_projector(fim, threshold=0.1)
    print(f"Projector shape: {projector.shape}")
    # Projector should be idempotent (P^2 = P)
    p_squared = torch.matmul(projector, projector)
    assert torch.allclose(projector, p_squared, atol=1e-5)
    
    # 3. Cramér-Rao Lower Bound
    # Grad of a causal functional (e.g., path sensitivity)
    c_grad = torch.randn(10, 1)
    crlb = InformationGeometricAnalyzer.calculate_cramer_rao_bound(fim, c_grad)
    print(f"Cramér-Rao Lower Bound: {crlb}")
    assert crlb > 0
    
    # 4. GCV Tikhonov
    # Singular FIM (one zero eigenvalue)
    fim_sing = torch.diag(torch.tensor([1.0, 1.0, 0.0]))
    c_grad_sing = torch.tensor([[1.0], [1.0], [1.0]])
    
    alpha_opt = MathematicalRigor.calculate_gcv_optimal_alpha(fim_sing, c_grad_sing)
    print(f"Optimal Alpha (GCV): {alpha_opt}")
    assert alpha_opt > 0
    
    tikhonov_crlb = MathematicalRigor.calculate_tikhonov_crlb(fim_sing, c_grad_sing)
    print(f"Tikhonov CRLB: {tikhonov_crlb}")
    assert tikhonov_crlb > 0
    
    # 5. Interpretability Uncertainty Principle
    # A: Semantic operator (diagonal), B: Causal operator (anti-diagonal)
    A = torch.diag(torch.tensor([1.0, 0.0]))
    B = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    
    # [A, B] = [[0, 1], [-1, 0]]
    # We need a state rho s.t. Tr(rho * [A, B]) is NOT zero.
    # [A, B] is anti-symmetric, so we need rho with an imaginary part (quantum)
    # OR we check the absolute commutator norm if we stay real-valued.
    # In our implementation, we use Tr(rho * [A, B]) which is sensitive to the state.
    
    # Let's use a state that has a non-zero trace with the commutator
    # For a real anti-symmetric matrix, Tr(rho * comm) is zero if rho is symmetric.
    # Let's verify the operator commutator logic separately and check the bound.
    
    # Complex state (or asymmetric proxy for testing the math flow)
    rho_asym = torch.tensor([[0.5, 0.8], [0.2, 0.5]]) # Not a valid density matrix but tests the math
    uncertainty_bound = InformationGeometricAnalyzer.calculate_interpretability_uncertainty(A, B, rho_asym)
    print(f"Asymmetric Uncertainty Bound: {uncertainty_bound}")
    # Tr(rho_asym * [[0, 1], [-1, 0]]) = Tr([[ -0.8, 0.5 ], [ -0.5, 0.2 ]]) = -0.6
    # Bound = 0.25 * |-0.6|^2 = 0.09
    assert uncertainty_bound == pytest.approx(0.09)
    
    print("✓ Information Geometry analysis validated")

def test_ricci_singularity():
    from src.analysis.scientific_rigor import MathematicalRigor
    print("\nTesting Mathematical Rigor: Ricci Flow & Singularities")
    
    # 1. Curvature Proxy
    # Low spread = Flat manifold
    fim_flat = torch.eye(10)
    curvature_flat = MathematicalRigor.calculate_scalar_curvature_proxy(fim_flat)
    print(f"Flat curvature: {curvature_flat}")
    assert curvature_flat < 1e-5
    
    # High spread = Curved manifold
    fim_curved = torch.diag(torch.tensor([100.0, 0.01, 1.0, 1.0]))
    curvature_curved = MathematicalRigor.calculate_scalar_curvature_proxy(fim_curved)
    print(f"High curvature: {curvature_curved}")
    assert curvature_curved > 1.0
    
    # 2. Singularity Detection
    # Normal training (smooth curvature growth)
    history_smooth = [1.0, 1.1, 1.2, 1.3, 1.4]
    result_smooth = MathematicalRigor.detect_geometric_singularities(history_smooth)
    print(f"Smooth flow singularity: {result_smooth['singularity_detected']}")
    assert not result_smooth["singularity_detected"]
    
    # Circuit Emergence (Sudden curvature spike / blow-up)
    # Long stable history with a massive sudden spike at the end
    history_spike = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]
    result_spike = MathematicalRigor.detect_geometric_singularities(history_spike)
    print(f"Spike flow singularity: {result_spike['singularity_detected']}")
    assert result_spike["singularity_detected"]
    
    print("✓ Ricci Flow & Singularity detection validated")

def test_rg_flow():
    from src.analysis.scientific_rigor import RGFlowAnalyzer
    print("\nTesting RG Flow Analyzer")
    lambda_history = [0.1, 0.08, 0.065, 0.055, 0.048]
    sigma_history = [0.5, 0.6, 0.65, 0.68, 0.7]
    result = RGFlowAnalyzer.estimate_beta_function(lambda_history, sigma_history)
    print(f"RG Flow beta function: {result}")
    assert "fixed_points" in result
    print("✓ RG Flow analysis validated")

def test_topos_analyzer():
    from src.analysis.scientific_rigor import ToposAnalyzer
    import torch
    print("\nTesting Topos Analyzer (Eilenberg-Moore Fixed Points)")
    
    # Mock encoder/decoder for a simple contraction mapping
    def mock_encoder(x): return x * 0.5
    def mock_decoder(x): return x * 0.5
    
    initial_state = torch.tensor([1.0, 2.0])
    result = ToposAnalyzer.find_eilenberg_moore_fixed_points(initial_state, mock_encoder, mock_decoder)
    print(f"Eilenberg-Moore Fixed Point result: converged={result['is_converged']}, iter={len(result['convergence_history'])}")
    assert result["is_converged"]
    assert torch.allclose(result["stable_state"], torch.zeros_like(initial_state), atol=1e-3)
    print("✓ Topos Analyzer validated")

def test_quantum_cognition():
    from src.analysis.scientific_rigor import QuantumCognitionAnalyzer
    import torch
    import numpy as np
    print("\nTesting Quantum Cognition Metrics")
    
    # 1. Wigner Negativity (PPT proxy)
    # Separable state (Identity)
    rho_sep = torch.eye(4) / 4.0
    neg_sep = QuantumCognitionAnalyzer.calculate_wigner_negativity(rho_sep)
    print(f"Separable Negativity: {neg_sep}")
    assert neg_sep < 1e-5
    
    # Bell state (Maximally entangled) -> High negativity
    bell = torch.tensor([[1.0, 0, 0, 1.0]]) / np.sqrt(2)
    rho_bell = torch.matmul(bell.t(), bell)
    neg_bell = QuantumCognitionAnalyzer.calculate_wigner_negativity(rho_bell)
    print(f"Bell state Negativity: {neg_bell}")
    assert neg_bell > 0.4 # Usually 0.5 for maximally entangled qubit pair
    
    # 2. Chern-Simons Holonomy
    # A_mu = 0 -> Wilson loop = I -> Trace = dim
    A_0 = torch.zeros(2, 2)
    holonomy_trivial = QuantumCognitionAnalyzer.calculate_chern_simons_holonomy([A_0, A_0])
    print(f"Trivial Holonomy Trace: {holonomy_trivial}")
    assert holonomy_trivial == pytest.approx(2.0)
    
    print("✓ Quantum Cognition Metrics validated")


if __name__ == "__main__":
    test_mathematical_rigor_coherence()
    test_mathematical_rigor_stability()
    test_sparsity_bound()
    test_advanced_spectral_rigor()
    test_topological_circuit_analyzer()
    test_operator_commutator()
    test_entanglement_entropy()
    test_lyapunov_stability()
    test_symmetry_noether()
    test_information_geometry()
    test_ricci_singularity()
    test_rg_flow()
    test_topos_analyzer()
    test_quantum_cognition()
    print("\nAll mathematical rigor tests PASSED!")
