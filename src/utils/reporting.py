import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class ScientificReportGenerator:
    """
    Generates human-readable scientific reports from NeuronMap's advanced mathematical audits.
    Translates complex metrics like Betti numbers and RG-flow into plain English/Math.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("outputs/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(self, audit_results: Dict[str, Any], circuit_id: str = "latest") -> str:
        """Creates a comprehensive Markdown report of the audit."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = []
        report.append(f"# NeuronMap Scientific Audit Report: {circuit_id}")
        report.append(f"*Generated on: {timestamp}*")
        report.append("\n## 1. Executive Summary")
        
        # Stability Assessment
        rg = audit_results.get('rg_flow', {})
        is_stable = rg.get('is_massive_phase', False)
        is_trivial = rg.get('is_ir_trivial', False)
        
        status = "STABLE (Massive Phase)" if is_stable else "UNSTABLE (Massless Phase)"
        if is_trivial:
            status += " - WARNING: IR TRIVIAL"
            
        report.append(f"- **Circuit Stability:** {status}")
        
        # Interpretability Guardrail
        negativity = audit_results.get('wigner_negativity', 0.0)
        report.append(f"- **Quantum Contextuality (Wigner Negativity):** {negativity:.4f}")
        
        # Topology
        topos = audit_results.get('topological_conflict', {})
        betti = topos.get('betti_numbers', {})
        p_val = audit_results.get('topological_p_value', 1.0)
        report.append(f"- **Topological Integrity:** {'CONFLICT DETECTED' if topos.get('has_topological_conflict') else 'INTEGRATED'}")
        report.append(f"- **Topological Significance (p-value):** {p_val:.4f} ({'Significant' if p_val < 0.05 else 'Potential Sampling Artifact'})")
        
        report.append("\n## 2. Advanced Mathematical Details")
        
        # RG-Flow Section
        report.append("\n### Renormalization Group (RG) Flow")
        report.append(f"The beta function $\\beta_\\lambda$ indicates the fixed-point structure of the latent field.")
        if is_trivial:
            report.append("> **WARNING:** The flow vanishes only at the Gaussian fixed point ($\\lambda^*=0$). This suggests the 'circuit' may be a transient fluctuation rather than a persistent world-model structure.")
        report.append(f"- **Fixed Points:** {rg.get('fixed_points', [])}")
        report.append(f"- **C1 (Self-Interaction):** {rg.get('c1', 0.0):.4f}")
        report.append(f"- **Interpretation:** {self._interpret_rg_flow(rg)}")
        
        # Topology Section
        report.append("\n### Simplicial Homology (H^k)")
        report.append(f"The circuit's latent geometry was triangulated to compute the Betti numbers.")
        report.append(f"- **$\\beta_0$ (Components):** {betti.get('beta_0', 0)}")
        report.append(f"- **$\\beta_1$ (Cycles):** {betti.get('beta_1', 0)}")
        report.append(f"- **$\\beta_2$ (Cavities/Conflicts):** {betti.get('beta_2', 0)}")
        report.append(f"- **Euler Characteristic ($\\chi$):** {topos.get('euler_characteristic', 0)}")
        
        # Quantum Section
        report.append("\n### Quantum Cognition & Contextuality")
        report.append(f"Measured non-classical correlations in the attention head compositions.")
        report.append(f"- **Wigner Negativity:** {negativity:.4f}")
        report.append(f"- **Chern-Simons Holonomy (Wilson Trace):** {audit_results.get('causal_holonomy_trace', 1.0):.4f}")
        report.append(f"- **Meaning:** {self._interpret_wigner(negativity)}")
        
        # Monadic Stability Section
        ms = audit_results.get('monadic_stability', {})
        report.append("\n### Monadic Fixed Points (Eilenberg-Moore)")
        report.append(f"Analyzed the convergence of the semantic mapping $T = R \\circ L$.")
        report.append(f"- **Monadic Stability:** {'CONVERGED' if ms.get('is_converged') else 'STOCHASTIC/UNSTABLE'}")
        report.append(f"- **Convergence Iterations:** {len(ms.get('convergence_history', []))}")
        
        # Visualizations
        vis_path = audit_results.get('topos_visualization_path')
        if vis_path:
            report.append(f"\n## 3. Visualizations")
            report.append(f"- [Interactive Topos-Logic Lattice (Heyting Algebra)]({vis_path})")
            
        report.append("\n---")
        report.append("*NeuronMap: Rigorous Mechanistic Interpretability Framework*")
        
        full_markdown = "\n".join(report)
        
        # Save to file
        report_file = self.output_dir / f"audit_report_{circuit_id}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(full_markdown)
            
        return str(report_file)

    def _interpret_rg_flow(self, rg: Dict[str, Any]) -> str:
        if rg.get('is_massive_phase'):
            return "The circuit has reached an Infrared (IR) fixed point. Features are expected to be robust across different model scales."
        return "The circuit is in a massless/scale-invariant phase. Features may be transient and susceptible to noise."

    def _interpret_wigner(self, negativity: float) -> str:
        if negativity > 0.1:
            return "High negativity detected. This suggests 'Quantum' contextuality; the model's logic cannot be decomposed into independent Boolean gates."
        return "Low negativity. The circuit behaves mostly like a classical logical gate (Boolean CCC)."

if __name__ == "__main__":
    # Demo
    gen = ScientificReportGenerator()
    mock_audit = {
        "rg_flow": {"is_massive_phase": True, "fixed_points": [0.0, 0.12], "c1": 5.4},
        "wigner_negativity": 0.45,
        "topological_conflict": {
            "betti_numbers": {"beta_0": 1, "beta_1": 2, "beta_2": 1},
            "has_topological_conflict": True,
            "euler_characteristic": 0
        },
        "topos_visualization_path": "audit_topos_latest.html"
    }
    path = gen.generate_markdown_report(mock_audit, "demo_circuit_001")
    print(f"Report generated at: {path}")
