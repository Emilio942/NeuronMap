# NeuronMap Bug Report

## 1. Causal Tracing `KeyError`
- **Location:** `src/analysis/conceptual_analysis.py`, line 827 in `_get_activations`
- **Description:** The code attempts to return `activations[target_layer]` after a forward pass. However, if the forward hook was not triggered (e.g., because the layer is in a non-executed branch of the model), the `activations` dictionary will be empty or missing the key.
- **Impact:** Crashes the analysis with a `KeyError`.
- **Recommended Fix:** Check if `target_layer` exists in the `activations` dictionary before access. Raise a descriptive `ActivationExtractionError` if missing.

## 2. Orchestrator Caching Logic Error
- **Location:** `src/core/orchestrator.py`, line 105 in `_run_analysis_worker`
- **Description:** The cache key `f"analysis:{model_name}:{hash(str(input_data))}"` only considers the model and the input data. It ignores the `analysis_types` parameter.
- **Impact:** If a user first requests "statistical" analysis and later requests "performance" analysis for the same model and input, the second request will return the cached "statistical" results without performing the "performance" analysis.
- **Recommended Fix:** Include `analysis_types` (e.g., as a sorted, comma-separated string) in the cache key.

## 3. SwiReasoning Entropy History Leak
- **Location:** `src/guardian/policies.py`, line 133 in `_switch_mode`
- **Description:** When switching between 'explicit' and 'latent' modes, the `entropy_history` list is not cleared.
- **Impact:** The entropy trend (slope) calculation in the new mode will be skewed by the final tokens of the previous mode, leading to potentially incorrect switching decisions.
- **Recommended Fix:** Clear `self.entropy_history = []` when `_switch_mode` is called to ensure the trend is calculated only for the current mode.

## 4. Activation Extractor Dimensionality Issue
- **Location:** `src/analysis/activation_extractor.py`, line 286 in `_hook_fn`
- **Description:** For a 3D tensor output `[batch, seq, hidden]`, `output_tensor.mean(dim=0)` results in a `[seq, hidden]` tensor. If the extractor is expected to return a single vector per question, this is incorrect.
- **Impact:** Downstream components expecting a 1D vector `[hidden]` will receive a 2D tensor, causing shape mismatch errors.
- **Recommended Fix:** Use `output_tensor.mean(dim=(0, 1))` to aggregate over both batch and sequence dimensions if a single summary vector is desired.

## 5. Placeholder API File
- **Location:** `src/api/zoo_web_api.py`
- **Description:** The file exists but is 0 bytes.
- **Impact:** Indicates incomplete implementation; will cause `ImportError` if referenced by other modules.
- **Recommended Fix:** Implement the necessary API endpoints or remove the file if it's no longer needed.

## 6. REST API Result Storage Inconsistency
- **Location:** `src/api/rest_api.py`
- **Description:** `self.results` is initialized in `__init__` but never updated in `run_analysis` or other methods. `get_analysis_results` (line 343) attempts to return data from `self.results`.
- **Impact:** The `/results/{analysis_id}` endpoint will always return a 404 error even for successfully completed jobs.
- **Recommended Fix:** Update `self.results` when a job completes or proxy the request to the `orchestrator.project_manager`.
