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
    
    # 3. Fidelity
    fidelity_self = MathematicalRigor.calculate_interaction_fidelity(activations_mixed, activations_mixed)
    print(f"Self-Fidelity: {fidelity_self}")
    assert fidelity_self > 0 # Positive value
    
    print("✓ von Neumann Entanglement Entropy validated")

def test_lyapunov_stability():
    print("\nTesting Mathematical Rigor: Lyapunov Stability")
    
    # 1. Stable dynamics (Jacobian < I)
    # 10 layers, 64 dims
    dim = 64
    jacobians_stable = [0.9 * torch.eye(dim) for _ in range(10)]
    lambda_stable = MathematicalRigor.calculate_maximal_lyapunov_exponent(jacobians_stable)
    print(f"Stable lambda_max: {lambda_stable}")
    # ln(0.9) ~ -0.105
    assert lambda_stable < 0
    
    # 2. Chaotic/Divergent dynamics (Jacobian > I)
    jacobians_chaotic = [1.1 * torch.eye(dim) for _ in range(10)]
    lambda_chaotic = MathematicalRigor.calculate_maximal_lyapunov_exponent(jacobians_chaotic)
    print(f"Chaotic lambda_max: {lambda_chaotic}")
    # ln(1.1) ~ 0.095
    assert lambda_chaotic > 0
    
    # 3. Prediction Horizon
    # lambda = -0.1, delta_x0 = 1e-3, epsilon = 1.0
    # H = floor(1/0.1 * ln(1/1e-3)) = floor(10 * 6.9) = 69
    horizon = MathematicalRigor.calculate_prediction_horizon(lambda_max=-0.1, delta_x0=1e-3)
    print(f"Prediction Horizon: {horizon}")
    assert horizon >= 60
    
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
    
    print("✓ Information Geometry analysis validated")

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
    print("\nAll mathematical rigor tests PASSED!")
