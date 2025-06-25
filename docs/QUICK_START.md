# 🚀 NeuronMap - Quick Start Guide

## ⚡ Sofortiger Einstieg

### Installation
```bash
# In NeuronMap Verzeichnis
cd /path/to/NeuronMap
pip install -r requirements.txt
```

### Erste Schritte
```bash
# 1. Setup validieren
python main.py validate

# 2. Konfiguration anzeigen  
python main.py config

# 3. Erste Analyse starten
python main.py extract --model "your-model" --list-layers
```

## 🎯 Hauptfunktionen (Stand: 20.06.2025)

### 📊 Basis-Analyse
```bash
# Aktivierungen extrahieren
python main.py extract --model gpt2 --target-layer "transformer.h.0"

# Visualisierungen erstellen
python main.py visualize --methods pca tsne umap

# Interaktive Plots
python main.py interactive --port 8501
```

### 🧠 Erweiterte Analyse  
```bash
# Interpretability (CAV, Saliency)
python main.py interpret --analysis-type cav --concept-file concepts.txt

# Experimentelle Analyse (RSA, CKA)
python main.py experiment --analysis-type rsa --comparison-model bert-base

# Domain-spezifische Analyse
python main.py domain --analysis-type code --input-file code_samples.json
```

### 🔬 Konzeptuelle Analyse (NEU!)
```bash
# Concept Extraction
python main.py conceptual --analysis-type concepts --model gpt2 --input-file data.json

# Circuit Discovery  
python main.py conceptual --analysis-type circuits --model gpt2 --input-file data.json

# Causal Tracing
python main.py conceptual --analysis-type causal --model gpt2 --input-file data.json
```

### 🛡️ Ethics & Bias
```bash
# Bias-Analyse
python main.py ethics --model gpt2 --texts-file texts.txt --groups-file groups.txt

# Model Card generieren
python main.py ethics --model gpt2 --generate-card
```

## 📁 Dateiformate

### Eingabedaten
- **JSON**: `{"text": "Sample text", "label": "category"}`
- **JSONL**: Eine JSON-Zeile pro Datensatz
- **TXT**: Einfacher Text, eine Zeile pro Sample

### Ausgaben
- **Aktivierungen**: `.npz` NumPy Archives
- **Visualisierungen**: `.png`, `.html` (interaktiv)
- **Analysen**: `.json` Ergebnisberichte

## ⚙️ Konfiguration

### models.yaml
```yaml
models:
  gpt2:
    name: "gpt2"
    tokenizer: "gpt2"
    device: "auto"
  bert:
    name: "bert-base-uncased"
    tokenizer: "bert-base-uncased"
    device: "auto"
```

### experiments.yaml
```yaml
experiments:
  default:
    max_length: 512
    batch_size: 16
    num_samples: 1000
```

## 🔧 Troubleshooting

### Häufige Probleme
1. **Import-Fehler**: `pip install -r requirements.txt`
2. **CUDA-Probleme**: `export CUDA_VISIBLE_DEVICES=0`
3. **Memory-Fehler**: Batch-Size reduzieren
4. **Model nicht gefunden**: Pfad in `models.yaml` prüfen

### Debug-Modus
```bash
python main.py --log-level DEBUG <command>
```

## 📚 Dokumentation

- **README.md**: Projektübersicht
- **tutorials/**: Step-by-Step Anleitungen  
- **docs/**: API-Dokumentation
- **examples/**: Praktische Beispiele

## 🎯 22 Verfügbare Kommandos

1. **generate** - Fragen generieren
2. **extract** - Aktivierungen extrahieren
3. **visualize** - Statische Visualisierungen
4. **interactive** - Interaktive Plots
5. **pipeline** - Vollständige Pipeline
6. **validate** - Setup validieren
7. **config** - Konfiguration anzeigen
8. **monitor** - System überwachen
9. **errors** - Fehler anzeigen
10. **multi-extract** - Multi-Layer Extraktion
11. **analyze** - Erweiterte Analyse
12. **attention** - Attention-Analyse
13. **discover** - Model Discovery
14. **process** - Datenverarbeitung
15. **validate-data** - Datenvalidierung
16. **metadata** - Metadaten verwalten
17. **interpret** - Interpretability-Analyse
18. **experiment** - Experimentelle Analyse
19. **probe** - Probing Tasks
20. **domain** - Domain-spezifische Analyse
21. **ethics** - Ethics & Bias Analyse
22. **conceptual** - Konzeptuelle Analyse ⭐ NEU!

## 🏆 Status: VOLLSTÄNDIG FUNKTIONSFÄHIG

```
✅ Alle Tests bestanden (8/8)
✅ ConceptualAnalyzer funktional
✅ CLI vollständig implementiert
✅ Dokumentation vollständig
✅ Produktionsreif
```

## 🚀 Nächste Schritte

1. **Demo ausführen**: `./demo_conceptual.sh`
2. **Eigene Daten analysieren**: Daten in `data/` ablegen
3. **Konfiguration anpassen**: `configs/` bearbeiten
4. **Community beitreten**: GitHub Issues/Discussions

---

**NeuronMap - Das umfassendste Toolkit für Neural Network Analysis! 🧠✨**
