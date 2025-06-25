# NeuronMap Scripts

Dieses Verzeichnis enthält Utility-Scripts für das NeuronMap-Projekt.

## 📁 Struktur

```
scripts/
└── utilities/
    ├── cleanup_project.py    # Projekt-Aufräum-Script
    ├── run_tests.py         # Test-Runner
    ├── run_modern.py        # Moderne Ausführung
    ├── run_comprehensive_tests.py  # Umfassende Tests
    ├── setup.sh             # Setup-Script
    └── start_web.py         # Web-Server-Start
```

## 🧹 cleanup_project.py

Räumt automatisch das Projekt auf und organisiert Dateien:

```bash
# Projekt aufräumen
python scripts/utilities/cleanup_project.py

# Oder direkt ausführen
./scripts/utilities/cleanup_project.py
```

**Was wird aufgeräumt:**
- Status-Reports → `archive/status_reports/`
- Validierungs-Scripts → `archive/validation_scripts/`
- Debug-Scripts → `archive/debug_scripts/`
- Demo-Dateien → `demos/`
- Datendateien → `data/`
- Log-Dateien → `logs/`
- Test-Dateien → `archive/temp_files/`
- Leere `__pycache__` Ordner werden entfernt

## 🚀 Weitere Scripts

### run_tests.py
```bash
python scripts/utilities/run_tests.py
```

### run_comprehensive_tests.py
```bash
python scripts/utilities/run_comprehensive_tests.py
```

### start_web.py
```bash
python scripts/utilities/start_web.py
```

## 📋 Automatisches Cleanup

Führen Sie regelmäßig das Cleanup-Script aus, um das Projekt sauber zu halten:

```bash
# Wöchentlich
python scripts/utilities/cleanup_project.py
```
