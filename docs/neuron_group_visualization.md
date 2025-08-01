# Neuron Group Visualization System

## Überblick

Das Neuron Group Visualization System ist eine erweiterte Komponente von NeuronMap, die speziell für die **Identifikation und Visualisierung von Neuron-Gruppen** entwickelt wurde, die zusammen während des Lernprozesses aktiviert werden. Das System erkennt funktionale Cluster von Neuronen und analysiert ihre Lernmuster.

## Hauptfunktionen

### 🧠 Neuron-Gruppenerkennung
- **Korrelationsbasiertes Clustering**: Identifiziert Neuronen, die konsistent zusammen aktiviert werden
- **K-Means Clustering**: Gruppiert Neuronen basierend auf Aktivierungsmustern
- **Hierarchisches Clustering**: Erstellt hierarchische Neuron-Gruppierungen
- **Dynamische Gruppengröße**: Automatische Anpassung der Gruppenanzahl

### 📊 Lernmuster-Analyse
- **Temporal Learning Events**: Erkennt spezifische Lernereignisse im Zeitverlauf
- **Skill Kategorisierung**: Klassifiziert Lernmuster nach Fähigkeitstypen:
  - Mathematische Fähigkeiten
  - Sprachliche Fähigkeiten  
  - Logische Fähigkeiten
  - Gedächtnisfähigkeiten
- **Lernstärke-Bewertung**: Quantifiziert die Intensität von Lernprozessen

### 🎨 Erweiterte Visualisierungen
- **Gruppen-Heatmaps**: Zeigt Neuron-Gruppen und ihre Aktivierungsmuster
- **Netzwerk-Visualisierungen**: Interaktive Darstellung von Gruppen-Interaktionen
- **Scatter-Plots**: 2D-Projektionen der Aktivierungsräume
- **Interaktive Dashboards**: Umfassende, interaktive Analyse-Oberflächen

## Installation und Verwendung

### Voraussetzungen

```bash
# Basis-Abhängigkeiten
pip install numpy pandas

# Für Visualisierungen
pip install matplotlib seaborn plotly

# Für erweiterte Analyse
pip install scikit-learn networkx

# Für interaktive Komponenten
pip install jupyter ipywidgets
```

### Grundlegende Verwendung

```python
from src.visualization import (
    NeuronGroupVisualizer, 
    create_neuron_group_analysis
)
import numpy as np
import pandas as pd

# Beispiel-Aktivierungsmatrix laden
activation_matrix = np.random.random((100, 50))  # 100 Samples, 50 Neuronen

# Metadaten für Fragen/Tasks (optional)
question_metadata = pd.DataFrame({
    'question': [f'Question {i}' for i in range(100)],
    'category': np.random.choice(['math', 'language', 'logic'], 100)
})

# Komplette Analyse ausführen
results = create_neuron_group_analysis(
    activation_matrix=activation_matrix,
    question_metadata=question_metadata,
    output_dir="outputs/neuron_groups"
)

print(f"Gefundene Neuron-Gruppen: {results['summary']['total_groups']}")
print(f"Identifizierte Lernereignisse: {results['summary']['total_learning_events']}")
```

### Erweiterte Verwendung

```python
# Detaillierte Kontrolle über den Analyseprozess
visualizer = NeuronGroupVisualizer(output_dir="outputs/detailed_analysis")

# 1. Neuron-Gruppen identifizieren
neuron_groups = visualizer.identify_neuron_groups(
    activation_matrix,
    method='correlation_clustering',  # oder 'kmeans', 'hierarchical'
    correlation_threshold=0.7,
    min_group_size=4
)

# 2. Lernmuster analysieren
learning_events = visualizer.analyze_learning_patterns(
    activation_matrix, 
    neuron_groups, 
    question_metadata
)

# 3. Spezifische Visualisierungen erstellen
heatmap_path = visualizer.visualize_neuron_groups(
    activation_matrix, neuron_groups, method='heatmap'
)

network_path = visualizer.visualize_neuron_groups(
    activation_matrix, neuron_groups, method='network'
)

# 4. Interaktives Dashboard erstellen
dashboard_path = visualizer.create_interactive_group_dashboard(
    activation_matrix, neuron_groups, learning_events, question_metadata
)

# 5. Analyse-Bericht generieren
report_path = visualizer.generate_group_analysis_report(
    activation_matrix, neuron_groups, learning_events
)
```

## Integration in bestehende NeuronMap-Workflows

```python
from src.visualization.enhanced_analysis import EnhancedAnalysisWorkflow

# Erweiterten Workflow initialisieren
workflow = EnhancedAnalysisWorkflow(config=your_config)

# Komplette Analyse mit traditionellen und Gruppen-Methoden
results = workflow.run_complete_analysis(
    activation_data={
        'activations': {'layer1': activation_matrix},
        'metadata': question_metadata
    },
    include_neuron_groups=True,
    output_dir="outputs/complete_analysis"
)

# Ergebnisse enthalten sowohl traditionelle als auch Gruppen-Analysen
print("Traditionelle Analyse:", results['traditional_analysis'])
print("Neuron-Gruppen Analyse:", results['neuron_group_analysis'])
print("Kombinierte Erkenntnisse:", results['combined_insights'])
```

## Output-Dateien und Strukturen

Das System generiert eine strukturierte Ausgabe:

```
outputs/neuron_groups/
├── visualizations/
│   ├── neuron_groups_heatmap.png          # Gruppen-Heatmap
│   ├── neuron_groups_network.png          # Netzwerk-Visualisierung
│   ├── neuron_groups_scatter.png          # Scatter-Plot
│   └── interactive_group_dashboard.html   # Interaktives Dashboard
├── reports/
│   ├── neuron_group_analysis_report.json  # Detaillierter JSON-Bericht
│   └── neuron_group_analysis_report.txt   # Lesbare Zusammenfassung
└── data/
    ├── neuron_groups.pkl                  # Serialisierte Gruppen-Objekte
    └── learning_events.json               # Lernerereignis-Daten
```

## Konfiguration und Parameter

### Neuron-Gruppenerkennung

```python
# Korrelationsbasiertes Clustering
neuron_groups = visualizer.identify_neuron_groups(
    activation_matrix,
    method='correlation_clustering',
    correlation_threshold=0.6,      # Korrelationsschwelle (0.0 - 1.0)
    min_group_size=3               # Minimale Gruppengröße
)

# K-Means Clustering  
neuron_groups = visualizer.identify_neuron_groups(
    activation_matrix,
    method='kmeans',
    n_groups=5,                    # Anzahl der Gruppen
    min_group_size=3
)

# Hierarchisches Clustering
neuron_groups = visualizer.identify_neuron_groups(
    activation_matrix,
    method='hierarchical',
    n_groups=4,
    min_group_size=2
)
```

### Visualisierungsoptionen

```python
# Farbschemata anpassen
visualizer.color_schemes['custom'] = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#F9CA24'
]

# Visualisierungsparameter
visualizer._setup_plotting_style()  # Setzt Standard-Stil

# Spezifische Visualisierung mit Parametern
heatmap_path = visualizer._visualize_groups_heatmap(
    activation_matrix, 
    neuron_groups,
    max_samples=50,     # Anzahl der angezeigten Samples
    color_scheme='custom'
)
```

## Demo und Beispiele

Ein vollständiges Demo-Skript ist verfügbar:

```bash
# Demo ausführen
python scripts/demo_neuron_groups.py

# Generiert Beispiel-Daten und zeigt alle Funktionen
# Outputs werden in 'demo_outputs/' gespeichert
```

Das Demo zeigt:
- Grundlegende Gruppenerkennung
- Lernmuster-Analyse
- Verschiedene Visualisierungsmethoden
- Interaktive Dashboard-Erstellung
- Integration mit echten NeuronMap-Daten

## Wissenschaftliche Grundlagen

### Korrelationsbasiertes Clustering

Das System verwendet Pearson-Korrelation zur Identifikation von Neuron-Gruppen:

```
correlation(i,j) = Σ((x_i - μ_i)(x_j - μ_j)) / √(Σ(x_i - μ_i)²Σ(x_j - μ_j)²)
```

Neuronen mit Korrelationen > Schwellenwert werden gruppiert.

### Kohäsions-Score

Die Gruppenkohäsion wird berechnet als:

```
cohesion = mean(correlation_matrix[upper_triangle])
```

Höhere Werte bedeuten stärkere interne Gruppenkohäsion.

### Lernstärke-Bewertung

Lernstärke basiert auf der maximalen Gruppenaktivierung:

```
learning_strength = max(mean_group_activations)
```

## Anwendungsfälle

### 1. **Modell-Interpretabilität**
- Verstehen, welche Neuronen zusammenarbeiten
- Identifikation funktionaler Spezialisierung
- Analyse von Lernprogression

### 2. **Modell-Optimierung**
- Informierte Pruning-Strategien
- Architektur-Verbesserungen basierend auf Gruppierungen
- Transfer Learning Guidance

### 3. **Forschung und Entwicklung**
- Vergleich verschiedener Modellarchitekturen
- Analyse von Lernalgorithmen
- Evaluation von Training-Strategien

## Fehlerbehebung

### Häufige Probleme

1. **ImportError bei Visualisierungsbibliotheken**
   ```bash
   pip install matplotlib seaborn plotly networkx
   ```

2. **Leere Neuron-Gruppen**
   - Korrelationsschwelle reduzieren (z.B. 0.5 statt 0.7)
   - Minimale Gruppengröße verringern
   - Aktivierungsmatrix auf NaN/Inf prüfen

3. **Langsame Performance**
   - Datenmatrix auf kritische Samples reduzieren
   - Korrelationsberechnung optimieren
   - Parallelisierung aktivieren

### Debug-Modus

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detaillierte Logs während der Analyse
visualizer = NeuronGroupVisualizer(output_dir="debug_output")
```

## Erweiterte Funktionen

### Custom Skill-Klassifikation

```python
def custom_skill_classifier(question_text: str) -> str:
    """Benutzerdefinierte Skill-Klassifikation"""
    if 'calculate' in question_text.lower():
        return 'mathematical'
    elif 'explain' in question_text.lower():
        return 'linguistic'
    else:
        return 'general'

# Anwendung in der Analyse
learning_events = visualizer.analyze_learning_patterns(
    activation_matrix, 
    neuron_groups, 
    question_metadata,
    skill_classifier=custom_skill_classifier
)
```

### Zeitreihen-Analyse

```python
# Bei sequenziellen Daten
temporal_analysis = visualizer.analyze_temporal_learning_patterns(
    activation_matrix,
    neuron_groups,
    time_windows=10  # Analyse in 10er-Zeitfenstern
)
```

## API-Referenz

### Klassen

- **`NeuronGroupVisualizer`**: Hauptklasse für Gruppen-Visualisierung
- **`NeuronGroup`**: Datenstruktur für Neuron-Gruppen
- **`LearningEvent`**: Datenstruktur für Lernerereignisse
- **`EnhancedAnalysisWorkflow`**: Integrierter Analyse-Workflow

### Funktionen

- **`create_neuron_group_analysis()`**: Komplette Analyse in einem Aufruf
- **`integrate_neuron_group_analysis()`**: Integration in bestehende Workflows

### Parameter

Siehe Docstrings in den entsprechenden Modulen für detaillierte Parameter-Beschreibungen.

## Contribution und Weiterentwicklung

Das Neuron Group Visualization System ist erweiterbar. Mögliche Verbesserungen:

1. **Erweiterte Clustering-Algorithmen** (DBSCAN, Spectral Clustering)
2. **Deep Learning-basierte Gruppenerkennung**
3. **Echtzeit-Analyse-Capabilities**
4. **Integration mit anderen ML-Frameworks**

---

Für weitere Fragen oder Unterstützung, siehe die Demo-Skripte oder erstellen Sie ein Issue im Repository.
