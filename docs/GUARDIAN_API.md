# Guardian Network API Documentation

This document describes the API endpoints for controlling the Guardian Network (Meta-cognitive Layer).

## Base URL
`/api/guardian`

## Endpoints

### 1. Get Configuration
Retrieves the current configuration of the Guardian Network.

- **URL**: `/config`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "enabled": true,
    "mode": "intervention",
    "intervention_layers": [5, 6],
    "entropy_min": 0.5,
    "entropy_max": 2.5,
    "noise_std": 0.1,
    "steering_coeff": 1.0
  }
  ```

### 2. Update Configuration
Updates the Guardian Network configuration at runtime.

- **URL**: `/config`
- **Method**: `POST`
- **Body**:
  ```json
  {
    "enabled": true,
    "mode": "monitoring",
    "entropy_min": 0.8
  }
  ```
- **Response**:
  ```json
  {
    "status": "updated",
    "config": { ... }
  }
  ```

## WebSocket Data Stream
The Realtime Visualization Stream (`ws://localhost:8765`) now includes Guardian metrics in each frame.

### Frame Structure
```json
{
  "timestamp": 1234567890.123,
  "layer_idx": 5,
  "activations": [...],
  "guardian_metrics": {
    "entropy": 1.2,
    "l2_norm": 5.4,
    "sparsity": 0.1,
    "collapsed": 0.0
  },
  "guardian_action": "none" // or "inject_noise", "apply_steering"
}
```

## Frontend Integration Guide

1. **Status Display**: Use `guardian_metrics` from the WebSocket stream to display real-time gauges for Entropy and Stability.
2. **Action Log**: Display `guardian_action` when it changes (e.g., flash a "NOISE INJECTED" alert).
3. **Control Panel**: Create a settings panel that calls `POST /api/guardian/config` to toggle the Guardian or adjust thresholds.
