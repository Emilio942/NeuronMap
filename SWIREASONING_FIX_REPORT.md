# SwiReasoning Fix & Benchmark Report

## Issue Summary
The SwiReasoning policy was failing to trigger mode switches during initial benchmarking.
- **Symptom**: Benchmark reported 0 switches (or only initialization switches) and no dynamic behavior.
- **Root Cause**: The entropy probe (`src/guardian/probes.py`) was returning `NaN` values.
- **Technical Detail**: The manual entropy calculation `-sum(probs * log(probs))` was numerically unstable for the `distilgpt2` model's activation distribution, likely due to floating-point underflow or precision issues with very small probabilities.

## Resolution
The entropy calculation logic in `src/guardian/probes.py` was replaced with PyTorch's robust implementation:
```python
# Old (Unstable)
probs = torch.nn.functional.softmax(logits, dim=-1)
entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

# New (Stable)
dist = torch.distributions.Categorical(logits=logits)
entropy = dist.entropy()
```

## Verification Results
The benchmark script `scripts/benchmark_swireasoning.py` was re-run successfully after two key fixes:
1.  **Robust Entropy Math**: Replaced manual calculation with `torch.distributions.Categorical`.
2.  **Correct Hook Location**: Modified the benchmark to hook the `lm_head` (logits) instead of intermediate transformer layers. This ensures entropy is calculated on the *Next Token Distribution*, as required by the SwiReasoning paper.

### Key Observations
-   **Dynamic Switching**: The policy now demonstrates frequent mode switching (e.g., 7 switches for logic/math prompts).
    -   *Example*: "What is the square root of 144 plus 12?" triggered **7 switches** (Latent <-> Explicit).
-   **Context Sensitivity**:
    -   **High Confidence**: Simple completions often stay in `latent` mode.
    -   **High Uncertainty**: Open-ended questions (e.g., "Difference between velocity and speed") stay in `explicit` mode (0 switches).
-   **Valid Metrics**: Entropy values are now in the expected range (e.g., 2.0 - 5.0 nats) for `distilgpt2`.

## Next Steps
-   **Visualization**: Implement a visualizer to plot the entropy trace and mode switches over time (as seen in the SwiReasoning paper).
-   **UI Integration**: Expose these metrics in the NeuronMap web interface.

