# Bereinigungsvorschläge für die Codebasis

Dieses Dokument listet Dateien auf, die zur Löschung oder Überarbeitung vorgeschlagen werden, um die Codebasis zu bereinigen und Redundanzen zu entfernen.

---

## 1. Redundante CLI-Befehlsdatei

- **Datei zur Löschung:** `src/cli/circuits_commands_new.py`

- **Grund:**
  - Die Datei ist eine nahezu exakte Kopie von `src/cli/circuits_commands.py`.
  - Sie definiert dieselben Kommandozeilen-Befehle, was zu Konflikten und Verwirrung führt.
  - Eine Suche im gesamten Projekt hat ergeben, dass die `_new.py`-Version nirgendwo importiert oder verwendet wird. Es handelt sich daher um toten Code, der wahrscheinlich von einem unvollendeten Refactoring übrig geblieben ist.

---

## 2. Redundante und veraltete CLI-Einstiegspunkte

- **Dateien zur Löschung:**
  1. `neuronmap-cli.py`
  2. `neuronmap.py`

- **Grund:**
  - Es existieren drei verschiedene Einstiegspunkte für die Kommandozeile (`main.py`, `neuronmap-cli.py`, `neuronmap.py`), was die Projektstruktur unnötig verkompliziert.
  - **`main.py`** ist die modernste und umfassendste Implementierung, die das `click`-Framework nutzt und alle Funktionalitäten (`surgery`, `circuits`, `sae`, `zoo`) unter einer einheitlichen Oberfläche vereint. Es sollte der einzige Einstiegspunkt sein.
  - **`neuronmap-cli.py`** ist eine veraltete ("legacy") Implementierung, die `argparse` verwendet und deren Funktionalität bereits in `main.py` enthalten ist.
  - **`neuronmap.py`** ist ein minimaler Wrapper, der nur einen kleinen Teil der Funktionalität aufruft, die ebenfalls vollständig von `main.py` abgedeckt wird.
  - Die Entfernung dieser beiden Dateien würde die Architektur klarer machen und die Wartung vereinfachen.

---

## 3. Unstrukturierte Daten- und Ausgabedateien

- **Dateien zum Verschieben:**
  - `cli_validation_report.json`
  - `cognitive_coordination_analysis_results.json`
  - `demo_results.json`
  - `demo_validation_results.json`
  - `neuronmap_errors.jsonl`
  - `performance_metrics.db`
  - `test_output.json`
  - `zoo_users.json`

- **Zielverzeichnis:** `data/outputs/`

- **Grund:**
  - Diese Dateien sind keine Konfigurations- oder Quellcode-Dateien, sondern Ergebnis-, Log- oder Datendateien.
  - Sie im Hauptverzeichnis zu belassen, macht dieses unübersichtlich.
  - Das Projekt enthält bereits ein `data/`-Verzeichnis mit einem `outputs/`-Unterverzeichnis, das für genau diesen Zweck vorgesehen ist. Das Verschieben verbessert die Trennung von Code und Daten.
