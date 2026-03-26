# NeuronMap - Kritischer Fehler- und Schwachstellenbericht

In diesem Dokument werden die identifizierten Softwarefehler dokumentiert, analysiert und kategorisiert. Dies dient als Grundlage für die Fehlerbehebung und Systemstabilisierung.

---

## 1. Causal Tracing: Hook-Uncertainty (KeyError)
*   **Dateipfad:** `src/analysis/conceptual_analysis.py` (Zeile 827)
*   **Warum ist das ein Fehler?** Die Funktion `_get_activations` greift blind auf das Dictionary `activations[target_layer]` zu. Wenn der PyTorch-Forward-Hook jedoch nicht ausgelöst wurde (z.B. weil der Layer im aktuellen Rechenpfad des Modells nicht aktiv ist), existiert dieser Key nicht.
*   **Auswirkung:** Sofortiger Programmabsturz mit einem `KeyError`.
*   **Einordnung:** **Kategorie: Stabilität / Kritischer Laufzeitfehler.**
*   **Lösung:** Vor dem Zugriff prüfen, ob der Key existiert (`if target_layer in activations`), andernfalls eine spezifische `ActivationExtractionError` werfen.

## 2. Orchestrator: Ungenügende Cache-Differenzierung
*   **Dateipfad:** `src/core/orchestrator.py` (Zeile 105)
*   **Warum ist das ein Fehler?** Der Cache-Key wird nur aus `model_name` und `input_data` generiert. Der Parameter `analysis_types` wird ignoriert.
*   **Auswirkung:** Wenn ein Nutzer zuerst eine statistische Analyse anfordert und danach eine Performance-Analyse für dasselbe Modell/Input, liefert das System die (falschen) gecachten Statistik-Ergebnisse zurück, anstatt die neue Analyse durchzuführen.
*   **Einordnung:** **Kategorie: Logikfehler / Datenkonsistenz.**
*   **Lösung:** Die angeforderten `analysis_types` müssen (sortiert) Teil des Cache-Keys werden.

## 3. SwiReasoning: Entropie-Historien-Leak
*   **Dateipfad:** `src/guardian/policies.py` (Zeile 133)
*   **Warum ist das ein Fehler?** Die Liste `self.entropy_history` wird beim Wechsel des Reasoning-Modus (`_switch_mode`) nicht geleert.
*   **Auswirkung:** Die Trendberechnung (Slope/Steigung) für den neuen Modus basiert auf den letzten Werten des alten Modus. Dies führt zu Fehlentscheidungen beim Switching, da der Trend „verunreinigt“ ist.
*   **Einordnung:** **Kategorie: Algorithmische Korrektheit / Accuracy.**
*   **Lösung:** `self.entropy_history = []` innerhalb von `_switch_mode` aufrufen.

## 4. Activation Extractor: Shape-Mismatch (Aggregation)
*   **Dateipfad:** `src/analysis/activation_extractor.py` (Zeile 286)
*   **Warum ist das ein Fehler?** Bei 3D-Tensoren (`[Batch, Sequence, Hidden]`) wird nur über `dim=0` gemittelt. Das Ergebnis ist ein 2D-Tensor (`[Sequence, Hidden]`). Erwartet wird jedoch oft ein 1D-Vektor pro Frage.
*   **Auswirkung:** Nachfolgende Analyse-Module, die einen Vektor erwarten, stürzen ab oder liefern falsche Korrelationen, da sie mit Matrizen statt Vektoren arbeiten.
*   **Einordnung:** **Kategorie: Datenintegrität / Integrationsfehler.**
*   **Lösung:** Über beide Dimensionen mitteln: `output_tensor.mean(dim=(0, 1))`.

## 5. REST API: Fehlende Daten-Persistenz (In-Memory Gap)
*   **Dateipfad:** `src/api/rest_api.py` (Zeile 343)
*   **Warum ist das ein Fehler?** Die Variable `self.results` wird im Konstruktor erstellt, aber in der Funktion `run_analysis` niemals mit den tatsächlichen Ergebnissen gefüllt.
*   **Auswirkung:** Der API-Endpunkt `/results/{analysis_id}` liefert immer einen 404-Fehler, selbst wenn die Analyse im Hintergrund erfolgreich abgeschlossen wurde.
*   **Einordnung:** **Kategorie: Funktionaler Fehler / API-Design.**
*   **Lösung:** Nach Abschluss des Jobs im Orchestrator müssen die Ergebnisse in `self.results` oder einer Datenbank gespeichert werden.

## 6. System-Struktur: Refactoring-Altlasten
*   **Verzeichnis:** `src/web/`
*   **Warum ist das ein Fehler?** Es existieren drei parallele App-Dateien: `app.py`, `app_new.py` und `app_old.py`.
*   **Auswirkung:** Massive Verwirrung für Entwickler. Es ist unklar, welche Datei die aktuelle Logik enthält. Änderungen werden oft in der falschen Datei vorgenommen.
*   **Einordnung:** **Kategorie: Wartbarkeit / Technical Debt.**
*   **Lösung:** Zusammenführung der Features in eine einzige `app.py` und Archivierung der alten Versionen.

---
**Bericht erstellt am:** 9. März 2026
**Status:** Offen (Zur Bearbeitung vorgemerkt)
