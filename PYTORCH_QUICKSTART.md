# PyTorch Neuron Group Quickstart Guide

## 🚀 Schnellstart in 5 Minuten

### 1. Installation & Setup

```bash
# Virtuelle Umgebung aktivieren
source .venv/bin/activate

# PyTorch installieren (falls noch nicht vorhanden)
pip install torch torchvision torchaudio

# Andere Abhängigkeiten
pip install matplotlib seaborn plotly scikit-learn pandas numpy
```

### 2. Einfachster Start - Standalone Demo

```bash
# Sofort loslegen ohne Plugin-Komplexität
python scripts/standalone_pytorch_demo.py
```

**Was passiert:**
- ✅ Erstellt ein Test-Neuronales Netz (TestNet)
- ✅ Generiert strukturierte Testdaten mit Mustern
- ✅ Extrahiert Aktivierungen aus 3 Layern
- ✅ Identifiziert 9 Neuron-Gruppen
- ✅ Erstellt 3 Visualisierungen
- ✅ Generiert detaillierten JSON-Report

**Ausgabe:**
```
demo_outputs/standalone_pytorch/
├── activation_heatmap_fc1.png      # Aktivierungs-Heatmap
├── group_cohesion.png              # Gruppen-Kohäsion
├── groups_per_layer.png            # Gruppen pro Layer
└── standalone_analysis_report.json # Detaillierter Report
```

### 3. Eigenes Model analysieren

```python
#!/usr/bin/env python3
"""Analyse Ihres eigenen PyTorch Models"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scripts.standalone_pytorch_demo import StandalonePyTorchVisualizer

# IHR MODEL HIER
class YourModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# IHRE DATEN HIER
X = torch.randn(100, 10)  # 100 Samples, 10 Features
dataset = TensorDataset(X)
dataloader = DataLoader(dataset, batch_size=16)

# ANALYSE STARTEN
model = YourModel()
visualizer = StandalonePyTorchVisualizer(output_dir="my_analysis")

# Schritt 1: Aktivierungen extrahieren
activations = visualizer.extract_activations(
    model, dataloader, ['layer1', 'layer2']
)

# Schritt 2: Neuron-Gruppen finden
groups = {}
for layer_name, activation_tensor in activations.items():
    groups[layer_name] = visualizer.identify_groups(
        activation_tensor, layer_name, threshold=0.3
    )

# Schritt 3: Visualisierungen erstellen
plots = visualizer.create_visualizations(activations, groups)

# Schritt 4: Report generieren
report = visualizer.generate_report(activations, groups, [])

print(f"✅ Analyse abgeschlossen! {len(plots)} Visualisierungen erstellt.")
```

### 4. Parameter anpassen

```python
# Mehr/weniger Gruppen finden
groups = visualizer.identify_groups(
    activation_tensor, 
    layer_name,
    threshold=0.2,      # Niedriger = mehr Gruppen
    min_size=2          # Kleinere Gruppen erlauben
)

# Weniger Daten verwenden (für große Models)
activations = visualizer.extract_activations(
    model, dataloader, layer_names, max_batches=5
)
```

## 🎯 Typische Use Cases

### Use Case 1: Layer-Vergleich

```python
# Script: compare_layers.py
visualizer = StandalonePyTorchVisualizer()
activations = visualizer.extract_activations(model, dataloader, 
    ['conv1', 'conv2', 'fc1', 'fc2'])

for layer_name, activation_tensor in activations.items():
    groups = visualizer.identify_groups(activation_tensor, layer_name)
    print(f"{layer_name}: {len(groups)} Gruppen gefunden")
    
    for group in groups:
        print(f"  Gruppe {group.group_id}: {group.group_size} Neuronen, "
              f"Kohäsion: {group.cohesion_score:.3f}")
```

### Use Case 2: Training-Verlauf analysieren

```python
# Script: training_analysis.py
checkpoints = ['model_epoch_1.pth', 'model_epoch_10.pth', 'model_epoch_20.pth']

for i, checkpoint in enumerate(checkpoints):
    model.load_state_dict(torch.load(checkpoint))
    
    visualizer = StandalonePyTorchVisualizer(f"analysis_epoch_{i+1}")
    activations = visualizer.extract_activations(model, dataloader, ['fc1', 'fc2'])
    
    total_groups = 0
    for layer_name, activation_tensor in activations.items():
        groups = visualizer.identify_groups(activation_tensor, layer_name)
        total_groups += len(groups)
    
    print(f"Epoch {i+1}: {total_groups} Gruppen")
```

### Use Case 3: Model-Vergleich

```python
# Script: model_comparison.py
models = {
    'small_model': SmallNet(),
    'large_model': LargeNet(),
    'pretrained_model': torch.load('pretrained.pth')
}

results = {}
for name, model in models.items():
    visualizer = StandalonePyTorchVisualizer(f"comparison_{name}")
    activations = visualizer.extract_activations(model, dataloader, ['fc1'])
    groups = visualizer.identify_groups(activations['fc1'], 'fc1')
    
    results[name] = {
        'total_groups': len(groups),
        'avg_cohesion': sum(g.cohesion_score for g in groups) / len(groups),
        'avg_group_size': sum(g.group_size for g in groups) / len(groups)
    }

print("Model Comparison:")
for name, metrics in results.items():
    print(f"{name}: {metrics['total_groups']} Gruppen, "
          f"Ø Kohäsion: {metrics['avg_cohesion']:.3f}")
```

## 🔧 Troubleshooting

### Problem: "No groups found"
```python
# Lösung: Parameter anpassen
groups = visualizer.identify_groups(
    activation_tensor, layer_name,
    threshold=0.1,      # Sehr niedrig
    min_size=2          # Kleine Gruppen
)

# Debug: Korrelations-Matrix checken
import numpy as np
activations_np = activation_tensor.cpu().numpy()
corr_matrix = np.corrcoef(activations_np.T)
print(f"Korrelationen: {corr_matrix.min():.3f} bis {corr_matrix.max():.3f}")
```

### Problem: "CUDA out of memory"
```python
# Lösung 1: CPU verwenden
visualizer = StandalonePyTorchVisualizer()
# Device wird automatisch auf CPU gesetzt wenn CUDA voll

# Lösung 2: Weniger Batches
activations = visualizer.extract_activations(
    model, dataloader, layer_names, max_batches=3
)

# Lösung 3: Kleinere Batch-Size
dataloader = DataLoader(dataset, batch_size=8)  # Statt 32
```

### Problem: "Import errors"
```python
# Standalone Version verwenden - keine Plugin-Abhängigkeiten
# Alle Funktionen sind in standalone_pytorch_demo.py verfügbar
```

## 📊 Output verstehen

### Gruppen-Statistiken
```json
{
  "group_id": 0,
  "size": 9,                    // 9 Neuronen in der Gruppe
  "cohesion": 0.842,           // Hohe Kohäsion (gut)
  "mean_activation": 0.358     // Durchschnittliche Aktivierung
}
```

**Cohesion Score Interpretation:**
- `> 0.7`: Sehr starke Gruppe 🟢
- `0.3 - 0.7`: Moderate Gruppe 🟡  
- `< 0.3`: Schwache Gruppe 🔴
- `< 0`: Anti-korrelierte Neuronen ⚫

### Visualisierungen verstehen

1. **groups_per_layer.png**: Zeigt Anzahl Gruppen pro Layer
2. **group_cohesion.png**: Kohäsions-Scores aller Gruppen
3. **activation_heatmap_fc1.png**: Aktivierungsmuster reorganisiert nach Gruppen

## ⚡ Quick Commands

```bash
# Komplette Pipeline in einem Befehl
python -c "
from scripts.standalone_pytorch_demo import main
main()
"

# Nur Simulation (ohne PyTorch)
python scripts/pytorch_simulation_demo.py

# Mit eigenen Parametern
python -c "
from scripts.standalone_pytorch_demo import StandalonePyTorchVisualizer
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Quick test mit eigenen Daten
model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 3))
X = torch.randn(50, 5)
dataloader = DataLoader(TensorDataset(X), batch_size=10)

viz = StandalonePyTorchVisualizer('quick_test')
acts = viz.extract_activations(model, dataloader, ['0', '2'])  # Layer 0 und 2
groups = {name: viz.identify_groups(tensor, name) for name, tensor in acts.items()}
plots = viz.create_visualizations(acts, groups)
print(f'Fertig! {len(plots)} Plots erstellt.')
"
```

## 🎉 Nächste Schritte

1. **Eigene Daten testen**: Ersetzen Sie die Testdaten durch Ihre eigenen
2. **Parameter optimieren**: Experimentieren Sie mit `threshold` und `min_size`
3. **Erweiterte Features**: Schauen Sie sich `docs/pytorch_neuron_group_guide.md` an
4. **Integration**: Bauen Sie die Analyse in Ihr Training ein

---

**Hilfe benötigt?** 
- 📖 Vollständige Dokumentation: `docs/pytorch_neuron_group_guide.md`
- 🔍 Beispiele: `scripts/standalone_pytorch_demo.py`
- 🎮 Demo: `scripts/pytorch_simulation_demo.py`
