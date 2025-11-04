# Neuron Group Visualization - Schnellstart Guide

## Was ist das Neuron Group Visualization System?

Das **Neuron Group Visualization System** erweitert NeuronMap um die Fähigkeit, **Gruppen von Neuronen zu identifizieren und zu visualisieren**, die zusammen während des Lernprozesses aktiviert werden. Dies ermöglicht es, funktionale Spezialisierung und Lernmuster in neuronalen Netzwerken zu verstehen.

## 🚀 Schnellstart

### 1. Setup ausführen

```bash
# Dependencies installieren
python scripts/setup_neuron_groups.py

# Demo ausführen
python scripts/demo_neuron_groups.py
```

### 2. Basis-Verwendung

```python
from src.visualization import create_neuron_group_analysis
import numpy as np
import pandas as pd

# Ihre Aktivierungsdaten laden
activation_matrix = np.random.random((100, 50))  # 100 Samples, 50 Neuronen

# Analyse ausführen
results = create_neuron_group_analysis(
    activation_matrix=activation_matrix,
    output_dir="outputs/my_analysis"
)

print(f"Gefunden: {results['summary']['total_groups']} Neuron-Gruppen")
```

## 🧠 Kernfunktionen

### Neuron-Gruppenerkennung
- **Korrelationsbasiert**: Findet Neuronen, die konsistent zusammen aktiviert werden
- **K-Means**: Gruppiert basierend auf Aktivierungsmustern  
- **Hierarchisch**: Erstellt hierarchische Gruppierungen

### Lernmuster-Analyse
- **Temporal Events**: Erkennt Lernereignisse im Zeitverlauf
- **Skill-Kategorisierung**: Klassifiziert nach Fähigkeitstypen
- **Lernstärke**: Quantifiziert Lernintensität

### Erweiterte Visualisierungen
- **Gruppen-Heatmaps**: Zeigt Gruppierungen und Aktivierungen
- **Netzwerk-Diagramme**: Interaktive Gruppen-Interaktionen
- **Interaktive Dashboards**: Umfassende Analyse-Oberflächen

## 📊 Output-Beispiele

Das System generiert:

```
outputs/neuron_groups/
├── neuron_groups_heatmap.png        # Gruppen-Visualisierung
├── neuron_groups_network.png        # Netzwerk-Darstellung  
├── neuron_groups_scatter.png        # 2D-Projektionen
├── interactive_group_dashboard.html # Interaktives Dashboard
└── neuron_group_analysis_report.json # Detaillierter Bericht
```

## 🔧 Integration in bestehende Workflows

```python
from src.visualization.enhanced_analysis import EnhancedAnalysisWorkflow

# Erweiterten Workflow verwenden
workflow = EnhancedAnalysisWorkflow(config=your_config)

results = workflow.run_complete_analysis(
    activation_data={'activations': {'layer1': activation_matrix}},
    include_neuron_groups=True
)

# Kombiniert traditionelle + Gruppen-Analysen
```

## 🎯 Anwendungsfälle

1. **Modell-Interpretabilität**: Verstehen, welche Neuronen zusammenarbeiten
2. **Lernprogression**: Analysieren, wie sich Fähigkeiten entwickeln
3. **Modell-Optimierung**: Informierte Pruning- und Architektur-Entscheidungen
4. **Forschung**: Vergleich verschiedener Modelle und Lernalgorithmen

## 🔍 Beispiel-Erkenntnisse

Typische Erkenntnisse aus der Analyse:

- **Funktionale Spezialisierung**: "Gruppe 1 (12 Neuronen) spezialisiert sich auf mathematische Aufgaben"
- **Lernprogression**: "Sprachverständnis entwickelt sich in Phase 2 des Trainings"
- **Gruppen-Interaktionen**: "Mathematik- und Logik-Gruppen arbeiten bei komplexen Problemen zusammen"
- **Effizienz**: "85% der Neuronen sind in funktionalen Gruppen organisiert"

## 📋 Systemvoraussetzungen

**Basis-Dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

**Für Interaktivität:**
```bash  
pip install plotly networkx jupyter
```

**Oder alles auf einmal:**
```bash
python scripts/setup_neuron_groups.py
```

## 🔬 Wissenschaftlicher Hintergrund

Das System basiert auf etablierten Methoden:

- **Korrelationsanalyse**: Pearson-Korrelation zur Gruppenerkennung
- **Clustering-Algorithmen**: K-Means, hierarchisches und DBSCAN-Clustering  
- **Dimensionsreduktion**: PCA und t-SNE für Visualisierungen
- **Netzwerkanalyse**: Graph-basierte Interaktionsmodelle

## 📚 Weitere Ressourcen

- **Vollständige Dokumentation**: `docs/neuron_group_visualization.md`
- **Demo-Skript**: `scripts/demo_neuron_groups.py`
- **Beispiel-Notebooks**: `tutorials/neuron_group_analysis.ipynb`
- **API-Referenz**: Docstrings in den Modulen

## 🛠️ Troubleshooting

**Häufige Probleme:**

1. **Import-Fehler**: Dependencies mit `pip install` nachinstallieren
2. **Leere Gruppen**: Korrelationsschwelle reduzieren (z.B. 0.5)
3. **Performance**: Datenmatrix auf kritische Samples reduzieren

**Debug-Modus:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## ✨ Was macht das System besonders?

- **Plug & Play**: Einfache Integration in bestehende NeuronMap-Workflows
- **Flexibel**: Verschiedene Clustering-Methoden und Parameter
- **Umfassend**: Von Gruppenerkennung bis interaktive Dashboards
- **Wissenschaftlich fundiert**: Basiert auf etablierten ML-Methoden
- **Erweiterbar**: Modularer Aufbau für Custom-Funktionen

---

**Erste Schritte:**
1. `python scripts/setup_neuron_groups.py` ausführen
2. `python scripts/demo_neuron_groups.py` testen  
3. Eigene Daten analysieren mit `create_neuron_group_analysis()`

**Support:** Siehe Dokumentation oder erstellen Sie ein Issue im Repository.
