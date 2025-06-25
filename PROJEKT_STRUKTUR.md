# NeuronMap - Projekt-Übersicht

## 📁 Ordnerstruktur

```
NeuronMap/
├── 📂 src/                     # Hauptquellcode (modular strukturiert)
│   ├── 📂 analysis/            # Analyse-Module
│   ├── 📂 data_generation/     # Daten-Generierung
│   ├── 📂 visualization/       # Visualisierung
│   ├── 📂 utils/              # Utilities und Tools
│   ├── 📂 api/                # API-Schnittstellen
│   └── 📂 data_processing/    # Datenverarbeitung
│
├── 📂 docs/                    # Dokumentation
│   ├── ENHANCED_FEATURES_DOCUMENTATION.md
│   ├── FUTURE_ROADMAP.md
│   ├── MULTI_MODEL_GUIDE.md
│   ├── PROJECT_OVERVIEW.md
│   └── QUICK_START.md
│
├── 📂 scripts/                 # Utility-Scripts
│   └── 📂 utilities/          # Hilfsskripte
│
├── 📂 demos/                   # Demo- und Beispieldateien
├── 📂 tests/                   # Test-Suite
├── 📂 configs/                 # Konfigurationsdateien
├── 📂 data/                    # Daten und Ergebnisse
├── 📂 logs/                    # Log-Dateien
├── 📂 outputs/                 # Ausgabedateien
├── 📂 examples/                # Beispiele
├── 📂 tutorials/               # Tutorials
│
├── 📂 archive/                 # Archivierte Dateien
│   ├── 📂 status_reports/     # Alte Status-Reports
│   ├── 📂 validation_scripts/ # Alte Validierungs-Scripts
│   └── 📂 debug_scripts/      # Debug-Scripts
│
├── 📂 legacy_backup/           # Legacy-Code-Backup
├── 📂 checkpoints/             # Batch-Processing-Checkpoints
├── 📂 plugins/                 # Plugin-System
├── 📂 web/                     # Web-Interface
└── 📂 static/                  # Statische Dateien
```

## 🔧 Wichtige Dateien

- **README.md** - Hauptdokumentation
- **aufgabenliste.md** - Vollständiger Aufgaben- und Implementierungsplan
- **requirements.txt** - Python-Abhängigkeiten
- **pyproject.toml** - Projekt-Konfiguration
- **docker-compose.yml** - Docker-Setup

## � Repository-Information

- **GitHub Repository**: https://github.com/Emilio942/NeuronMap
- **Autor**: Emilio942
- **Lizenz**: MIT License
- **Sprache**: Python 3.8+

## �🚀 Schnellstart

1. **Repository klonen**:
1. **Repository klonen**:
   ```bash
   git clone https://github.com/Emilio942/NeuronMap.git
   cd NeuronMap
   ```

2. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Grundlegende Nutzung**:
   ```bash
   python -m src.analysis.activation_extractor --help
   python -m src.visualization.core_visualizer --help
   ```

3. **Tests ausführen**:
   ```bash
   cd scripts/utilities && python run_tests.py
   ```

## 📋 Status

- ✅ **Modularisierung**: Vollständig abgeschlossen
- ✅ **Input/Output-Validierung**: Implementiert
- ✅ **Quality Assurance**: Implementiert
- ✅ **Batch Processing**: Implementiert
- ✅ **Structured Logging**: Implementiert
- ✅ **Troubleshooting System**: Implementiert

**Status**: 🏆 **Production-Ready**

## 📚 Weitere Dokumentation

- [Erweiterte Features](docs/ENHANCED_FEATURES_DOCUMENTATION.md)
- [Multi-Model Guide](docs/MULTI_MODEL_GUIDE.md)
- [Schnellstart](docs/QUICK_START.md)
- [Zukunfts-Roadmap](docs/FUTURE_ROADMAP.md)
- [Projekt-Übersicht](docs/PROJECT_OVERVIEW.md)
