# INTEGRATION: Guardian Network ("Wächter-Modul")

## 1. Architektur-Merge: Der logische Schnittpunkt
Wir klinken das Guardian-Modul direkt in die `UniversalModelAdapter`-Klasse (`src/analysis/universal_model_adapter.py`) und den `ActivationExtractor` ein.

*   **Hook-Point**: Nutzung von PyTorch `register_forward_hook` auf den `TransformerBlock`-Ebenen.
*   **Intervention Layer**: Erweiterung der Hooks von "Read-Only" zu "Read-Write" (Modifikation des Output-Tensors).
*   **Neues Modul**: `src/guardian/` als "Man-in-the-Middle".

## 2. Datenfluss & Performance-Strategie
*   **Zero-Copy Strategy**: Wächter läuft auf demselben Device (GPU) wie das Hauptmodell.
*   **Detached Tensors**: Nutzung von `tensor.detach()` für die Analyse (kein Gradientenfluss).
*   **Synchroner Eingriff**: Blockierende Hooks für FlowRL-Steuerung.
*   **Optimierung**: Nutzung von SAEs oder MLP-Probes (<1ms Latenz).

## 3. Aufgabenliste (Step-by-Step)

### Phase 1: Vorbereitung & Refactoring (Infrastructure)
- [x] **Task 1.1: Config-Erweiterung**
    - Erweitere `src/utils/config_manager.py` um `GuardianConfig`.
    - Parameter: `intervention_layers`, `guardian_model_path`, `flow_thresholds`.
- [x] **Task 1.2: Refactor `ActivationExtractor`**
    - Erstelle `InterventionExtractor` (Subklasse).
    - Implementiere `_intervention_hook` mit Rückgabewert (modifizierter Tensor).
- [x] **Task 1.3: Modul-Struktur anlegen**
    - Erstelle `src/guardian/` mit `__init__.py`, `engine.py`, `probes.py`, `policies.py`, `interventions.py`.

### Phase 2: Implementierung der Probes (Sensoren)
- [x] **Task 2.1: GPU-Accelerated Metrics (`src/guardian/probes.py`)**
    - Implementiere `calculate_entropy(tensor)` (PyTorch native).
    - Implementiere `calculate_l2_norm(tensor)`.
    - Implementiere `detect_collapse(tensor)`.
- [x] **Task 2.2: Latent Space Projection**
    - Implementiere Projektion von $d_{model}$ auf Wächter-Input-Space.

### Phase 3: Der Wächter (Brain & Policy)
- [x] **Task 3.1: Guardian Model Loader (`src/guardian/engine.py`)**
    - Implementiere `GuardianEngine` zum Laden des sekundären Netzes.
    - Sicherstellen des korrekten Device-Managements.
- [x] **Task 3.2: Policy Implementation (`src/guardian/policies.py`)**
    - Implementiere `FlowPolicy` (Entscheidungslogik basierend auf Metriken).
    - Logic: Trigger für Noise Injection oder Steering basierend auf Entropie-Schwellenwerten.

### Phase 4: Schließen des Regelkreises (Intervention)
- [x] **Task 4.1: Intervention Primitives (`src/guardian/interventions.py`)**
    - `inject_gaussian_noise(tensor, std)`
    - `apply_steering_vector(tensor, vector, coeff)`
    - `ablate_neurons(tensor, mask)`
- [x] **Task 4.2: Integration in `UniversalModelAdapter`**
    - Modifiziere `load_model` zur Initialisierung der `GuardianEngine`.
    - Registrierung der `InterventionHooks` bei `guardian_enabled=True`.
- [x] **Task 4.3: Feedback-Loop Test**
    - Erstelle Integrationstest `tests/test_guardian_loop.py`.
    - Verifiziere Output-Änderung durch Wächter-Eingriff.

### Phase 5: GUI & Visualisierung (Frontend-Integration)
- [x] **Task 5.1: Update `ActivationFrame`**
    - Erweitere `src/visualization/realtime_streamer.py`, um Guardian-Metriken (Entropie, Status, Intervention) zu transportieren.
- [x] **Task 5.2: Update `RealtimeVisualizationEngine`**
    - Integriere Guardian-Status in den Datenstrom.
- [x] **Task 5.3: API-Endpunkte**
    - Erstelle Endpunkte in `src/web/app.py` für Guardian-Steuerung (Config-Update zur Laufzeit).
- [x] **Task 5.4: Dokumentation für Frontend**
    - Dokumentiere API-Nutzung für die GUI-Entwicklung.
