# NeuronMap: Dokumentation der „Zeit-Lösungen“ (Workarounds & Fallbacks)

In der aktuellen Codebasis von NeuronMap wurden verschiedene Mechanismen identifiziert, die als temporäre Lösungen, Ausweichmanöver (Fallbacks) oder zur Aufrechterhaltung des Betriebs bei fehlenden Ressourcen implementiert wurden.

## 1. Infrastruktur-Fallbacks (Resilienz)
Diese Mechanismen verhindern Systemabstürze bei fehlenden Abhängigkeiten oder Hardware-Ressourcen.

### Task-Queue Fallback
- **Datei:** `src/core/orchestrator.py`
- **Lösung:** Wenn die Initialisierung der angeforderten Queue (z. B. Redis) fehlschlägt, erfolgt ein automatischer, stiller Rückfall auf eine lokale Queue.
- **Code-Stelle:** Konstruktor von `SystemOrchestrator`.

### GPU-zu-CPU Recovery
- **Datei:** `src/utils/error_handling.py`
- **Lösung:** Der Dekorator `robust_execution` fängt `OutOfMemoryErrors` ab und versucht, die Operation auf der CPU fortzusetzen, falls die GPU-Ressourcen erschöpft sind.

### Modell-Fallback-Liste
- **Datei:** `src/utils/error_handling.py`
- **Lösung:** Die Klasse `AutomaticRecovery` enthält eine fest kodierte Mapping-Tabelle für Ersatzmodelle (z. B. `distilgpt2` als Ersatz für `gpt2`), falls das Hauptmodell nicht geladen werden kann.

## 2. Mock-Daten & Stubs
Diese Lösungen dienen als Platzhalter für noch nicht implementierte oder fehlende Komponenten.

### Aktivierungs-Mocking
- **Datei:** `src/core/orchestrator.py`
- **Lösung:** In `_extract_activations` werden zufällige NumPy-Arrays generiert, falls kein gültiger `ModelAdapter` gefunden wird. Dies erlaubt das Testen der Pipeline ohne echtes Modell.

### Core-Sentinel Objekt
- **Datei:** `src/core/neuron_map.py`
- **Lösung:** Ein minimaler "Stub"-Klasse `NeuronMap`, die nur existiert, um Import-Fehler in Web-UI-Prototypen zu vermeiden. Sie enthält keine echte Geschäftslogik.

### Leere API-Endpunkte
- **Datei:** `src/api/zoo_web_api.py`
- **Lösung:** Eine Datei mit 0 Bytes, die als Platzhalter für die zukünftige Zoo-Web-Integration dient.

## 3. Daten-Parsing & API Workarounds

### Robustes JSONL-Parsing
- **Datei:** `src/analysis/activation_extractor.py`
- **Lösung:** In `load_questions` wird bei einem `JSONDecodeError` die Zeile einfach als Klartext übernommen. Dies verhindert den Abbruch des Ladeprozesses bei leicht fehlerhaften Datensätzen.

### Hartkodierte Demo-Visualisierung
- **Datei:** `src/api/simple_server.py`
- **Lösung:** Ein kompletter Flask-Server, der HTML und JavaScript als Python-String enthält. Dies umgeht Probleme mit Dateipfaden und statischen Assets in einer Testumgebung.

### In-Memory Job-Tracking
- **Datei:** `src/api/rest_api.py`
- **Lösung:** Jobs werden nur in einem lokalen Dictionary (`self.jobs`) gespeichert. Bei einem Server-Neustart gehen alle Statusinformationen verloren, da keine persistente Datenbank (wie PostgreSQL) angebunden ist.

---
**Status:** Diese Dokumentation dient zur Identifizierung technischer Schulden und zur Planung zukünftiger Refactorings.
