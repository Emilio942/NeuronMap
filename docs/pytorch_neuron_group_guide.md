# PyTorch Neuron Group Visualization für NeuronMap

## Überblick

Die PyTorch-Integration von NeuronMap ermöglicht die Analyse und Visualisierung von Neuron-Gruppen direkt aus PyTorch-Modellen. Das System identifiziert Gruppen von Neuronen, die gemeinsam aktiviert werden, und analysiert deren Lernmuster.

## Features

### 🔥 PyTorch-Integration
- **Native PyTorch Support**: Direkter Import von Aktivierungen aus PyTorch-Modellen
- **GPU/CUDA Unterstützung**: Automatische Erkennung und Nutzung verfügbarer Hardware
- **Flexible Model Support**: Unterstützung für beliebige PyTorch nn.Module
- **Hook-basierte Extraktion**: Effiziente Aktivierungsextraktion ohne Modellmodifikation

### 🧩 Neuron Group Identification
- **Korrelations-basiertes Clustering**: Identifikation korrelierter Neuronen
- **K-means Clustering**: Alternative Gruppierungsmethode
- **Hierarchisches Clustering**: Erweiterte Gruppierungsoptionen
- **Konfigurierbare Parameter**: Anpassbare Schwellenwerte und Gruppengrößen

### 📊 Advanced Visualizations
- **Layer-spezifische Heatmaps**: Detaillierte Aktivierungsmuster pro Layer
- **Scatter Plots**: 2D-Darstellung der Gruppenverteilung mit PCA/t-SNE
- **Interactive Dashboards**: Web-basierte interaktive Analysen
- **Multi-Layer Visualizations**: Vergleichende Darstellung mehrerer Layer

### 📚 Learning Pattern Analysis
- **Learning Events**: Identifikation von Lern-relevanten Aktivierungsmustern
- **Skill Categorization**: Automatische Klassifikation von Fähigkeitstypen
- **Temporal Analysis**: Zeitliche Entwicklung der Lernmuster
- **Loss Integration**: Einbeziehung von Loss-Werten in die Analyse

## Schnellstart

### 1. Basic Usage

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.visualization.pytorch_neuron_group_visualizer import create_pytorch_neuron_group_analysis

# Ihr PyTorch Model
model = YourPyTorchModel()

# Ihre Daten
dataloader = DataLoader(your_dataset, batch_size=32)

# Layer die analysiert werden sollen
layer_names = ['conv1', 'fc1', 'fc2']

# Vollständige Analyse durchführen
results = create_pytorch_neuron_group_analysis(
    model=model,
    dataloader=dataloader,
    layer_names=layer_names,
    output_dir="outputs/pytorch_analysis"
)
```

### 2. Standalone Usage (ohne Plugin-System)

```python
# Für einfache Nutzung ohne Abhängigkeiten
from scripts.standalone_pytorch_demo import StandalonePyTorchVisualizer

visualizer = StandalonePyTorchVisualizer()

# Aktivierungen extrahieren
activations = visualizer.extract_activations(model, dataloader, layer_names)

# Gruppen identifizieren
groups = {}
for layer_name, activation_tensor in activations.items():
    groups[layer_name] = visualizer.identify_groups(activation_tensor, layer_name)

# Visualisierungen erstellen
plots = visualizer.create_visualizations(activations, groups)
```

## Detaillierte API

### PyTorchNeuronGroupVisualizer

```python
class PyTorchNeuronGroupVisualizer:
    def __init__(self, output_dir="outputs", device='auto'):
        """
        Args:
            output_dir: Ausgabeverzeichnis für Visualisierungen
            device: PyTorch device ('auto', 'cpu', 'cuda')
        """
    
    def extract_activations_from_model(self, model, dataloader, layer_names, max_batches=None):
        """
        Extrahiert Aktivierungen aus PyTorch Model.
        
        Args:
            model: PyTorch nn.Module
            dataloader: DataLoader mit Input-Daten
            layer_names: Liste der zu analysierenden Layer-Namen
            max_batches: Maximale Anzahl Batches (None für alle)
        
        Returns:
            Dict[str, torch.Tensor]: Layer-Name -> Aktivierungs-Tensor
        """
    
    def identify_neuron_groups_pytorch(self, activation_tensor, layer_name, method='correlation_clustering'):
        """
        Identifiziert Neuron-Gruppen.
        
        Args:
            activation_tensor: Tensor der Form (n_samples, n_neurons)
            layer_name: Name des Layers
            method: 'correlation_clustering', 'kmeans', 'hierarchical'
        
        Returns:
            List[PyTorchNeuronGroup]: Identifizierte Neuron-Gruppen
        """
```

### PyTorchNeuronGroup Dataclass

```python
@dataclass
class PyTorchNeuronGroup:
    group_id: int                    # Eindeutige Gruppen-ID
    neuron_indices: List[int]        # Indizes der Neuronen in der Gruppe
    layer_name: str                  # Name des Layers
    activation_pattern: torch.Tensor # Aktivierungsmuster der Gruppe
    group_center: torch.Tensor       # Zentrum der Gruppe
    group_size: int                  # Anzahl Neuronen in der Gruppe
    cohesion_score: float            # Kohäsions-Score (0-1)
    learning_phase: Optional[str]    # Lernphase (falls identifiziert)
    skill_category: Optional[str]    # Fähigkeits-Kategorie
    device: str                      # PyTorch Device
```

## Demo Scripts

### 1. Vollständiges PyTorch Demo
```bash
python scripts/pytorch_demo_neuron_groups.py
```
Zeigt die vollständige Integration mit komplexen Modellen und Daten.

### 2. Simulation Demo  
```bash
python scripts/pytorch_simulation_demo.py
```
Demonstriert Konzepte ohne PyTorch-Installation (NumPy-basiert).

### 3. Standalone Demo
```bash
python scripts/standalone_pytorch_demo.py
```
Einfache Implementation ohne Plugin-Abhängigkeiten.

## Konfiguration

### Clustering Parameter

```python
# Korrelations-basiertes Clustering
groups = visualizer.identify_neuron_groups_pytorch(
    activation_tensor,
    layer_name,
    method='correlation_clustering',
    correlation_threshold=0.6,    # Minimum Korrelation (0-1)
    min_group_size=3             # Minimum Neuronen pro Gruppe
)

# K-means Clustering
groups = visualizer.identify_neuron_groups_pytorch(
    activation_tensor,
    layer_name, 
    method='kmeans',
    n_groups=5,                  # Anzahl Gruppen (auto wenn None)
    min_group_size=3
)
```

### Visualization Options

```python
# Heatmap Visualization
heatmap_path = visualizer.visualize_pytorch_groups(
    activations_dict, groups_dict, method='heatmap'
)

# Scatter Plot mit PCA
scatter_path = visualizer.visualize_pytorch_groups(
    activations_dict, groups_dict, method='scatter', layer_name='fc1'
)

# Interactive Dashboard
dashboard_path = visualizer.create_pytorch_interactive_dashboard(
    activations_dict, groups_dict, learning_events, metadata
)
```

## Ausgabe-Dateien

Die PyTorch-Analyse generiert folgende Dateien:

### Visualisierungen
- `pytorch_groups_heatmap_[layer].png` - Layer-spezifische Heatmaps
- `pytorch_groups_scatter_[layer].png` - Scatter Plots mit Dimensionalitätsreduktion
- `pytorch_all_layers_heatmap.png` - Multi-Layer Übersicht
- `pytorch_interactive_dashboard.html` - Interactive Web-Dashboard

### Reports
- `pytorch_neuron_group_analysis_report.json` - Detaillierter JSON-Report
- `pytorch_neuron_group_analysis_report.txt` - Lesbarer Text-Report

### Beispiel Report-Struktur

```json
{
  "pytorch_analysis": {
    "device": "cuda",
    "torch_version": "2.7.1+cu126",
    "total_layers": 3,
    "analysis_timestamp": "2025-08-01T16:50:08.049668"
  },
  "layer_analysis": {
    "fc1": {
      "groups_found": 5,
      "activation_shape": [80, 16],
      "groups_detail": [
        {
          "group_id": 0,
          "size": 9,
          "cohesion": 0.842,
          "mean_activation": 0.358
        }
      ]
    }
  },
  "learning_patterns": {
    "total_events": 25,
    "skill_distribution": {
      "mathematical": 8,
      "linguistic": 7,
      "logical": 6,
      "memory": 4
    },
    "average_strength": 0.746
  }
}
```

## Best Practices

### 1. Model Preparation
```python
# Model in Evaluation Mode setzen
model.eval()

# Für konsistente Ergebnisse
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
```

### 2. Memory Management
```python
# Für große Modelle: Batch-Limit setzen
activations = visualizer.extract_activations_from_model(
    model, dataloader, layer_names, max_batches=20
)

# GPU Memory sparen
activations = {name: tensor.cpu() for name, tensor in activations.items()}
```

### 3. Layer Selection
```python
# Relevante Layer identifizieren
for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        print(f"Analysierbar: {name}")

# Empfohlene Layer-Typen
layer_types = [nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU]
```

## Troubleshooting

### Häufige Probleme

1. **Import Errors**: Nutze `standalone_pytorch_demo.py` für Plugin-freie Analyse
2. **CUDA Memory**: Reduziere `max_batches` oder nutze `device='cpu'`
3. **Keine Gruppen gefunden**: Senke `correlation_threshold` oder `min_group_size`
4. **Große Modelle**: Nutze `torch.no_grad()` und begrenzte Batch-Anzahl

### Debug-Tipps

```python
# Aktivierungs-Shapes überprüfen
for name, tensor in activations.items():
    print(f"{name}: {tensor.shape}")

# Korrelations-Matrix inspizieren
corr_matrix = torch.corrcoef(activation_tensor.T)
print(f"Korrelationen: {corr_matrix.min():.3f} - {corr_matrix.max():.3f}")

# Memory Usage
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated():.1f}MB")
```

## Integration in bestehende Workflows

### 1. Training Integration
```python
# Während Training
for epoch in range(num_epochs):
    # ... training code ...
    
    if epoch % 10 == 0:  # Alle 10 Epochen analysieren
        results = create_pytorch_neuron_group_analysis(
            model, val_dataloader, layer_names,
            model_epoch=epoch
        )
```

### 2. Model Comparison
```python
# Verschiedene Modelle vergleichen
models = {'model_a': model_a, 'model_b': model_b}

for name, model in models.items():
    results = create_pytorch_neuron_group_analysis(
        model, dataloader, layer_names,
        output_dir=f"outputs/{name}"
    )
```

## Erweiterte Features

### Custom Hook Functions
```python
# Custom Activation Hook
def custom_hook(name):
    def hook(module, input, output):
        # Custom processing
        processed = F.normalize(output, p=2, dim=1)
        activations[name].append(processed.detach().cpu())
    return hook
```

### Multi-GPU Support
```python
# DataParallel Models
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    
# Layer names für DataParallel
layer_names = ['module.fc1', 'module.fc2']  # 'module.' prefix
```

## Performance Optimierung

1. **Batch Processing**: Nutze größere Batches für bessere GPU-Auslastung
2. **Memory Efficiency**: Aktivierungen sofort auf CPU verschieben
3. **Layer Selection**: Nur relevante Layer analysieren
4. **Parallel Processing**: Multiple GPUs für verschiedene Layer

## Zukunft & Roadmap

- 🔄 **Streaming Analysis**: Real-time Analyse während Training
- 🎯 **Attention Mechanisms**: Spezielle Unterstützung für Transformer
- 📱 **Mobile Integration**: TorchScript/Mobile Compatibility  
- 🔗 **AutoML Integration**: Automatische Hyperparameter-Optimierung
- 🌐 **Distributed Training**: Multi-Node Support

---

*Letzte Aktualisierung: August 2025*
*Version: 1.0.0*
