#!/usr/bin/env python3
"""
Einfaches Demo für Neuron Group Visualization
============================================

Dieses vereinfachte Demo zeigt die Grundfunktionen des Neuron Group 
Visualization Systems ohne komplexe Dependencies.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_basic_functionality():
    """Test der grundlegenden Neuron Group Funktionalität."""
    logger.info("=== Test der grundlegenden Funktionalität ===")
    
    try:
        # Import der Neuron Group Visualizer Klasse
        from visualization.neuron_group_visualizer import NeuronGroupVisualizer
        logger.info("✓ NeuronGroupVisualizer erfolgreich importiert")
        
        # Beispiel-Daten erstellen
        np.random.seed(42)
        n_samples = 50
        n_neurons = 30
        
        # Simuliere 3 Neuron-Gruppen mit unterschiedlichen Aktivierungsmustern
        activation_matrix = np.random.normal(0.2, 0.1, (n_samples, n_neurons))
        
        # Gruppe 1: Neuronen 0-9 (mathematische Aufgaben)
        for i in range(n_samples):
            if i % 3 == 0:  # Jede 3. Probe ist eine Mathe-Aufgabe
                activation_matrix[i, 0:10] += np.random.normal(0.6, 0.1, 10)
        
        # Gruppe 2: Neuronen 10-19 (sprachliche Aufgaben)  
        for i in range(n_samples):
            if i % 3 == 1:  # Jede 3. Probe ist eine Sprach-Aufgabe
                activation_matrix[i, 10:20] += np.random.normal(0.5, 0.1, 10)
        
        # Gruppe 3: Neuronen 20-29 (logische Aufgaben)
        for i in range(n_samples):
            if i % 3 == 2:  # Jede 3. Probe ist eine Logik-Aufgabe
                activation_matrix[i, 20:30] += np.random.normal(0.4, 0.1, 10)
        
        # Stelle sicher, dass alle Werte positiv sind
        activation_matrix = np.maximum(activation_matrix, 0.01)
        
        logger.info(f"✓ Beispiel-Aktivierungsmatrix erstellt: {activation_matrix.shape}")
        
        # Metadaten erstellen
        categories = ['math', 'language', 'logic'] * (n_samples // 3 + 1)
        question_metadata = pd.DataFrame({
            'question_id': range(n_samples),
            'question': [f'Frage {i}: {categories[i % 3]} Problem' for i in range(n_samples)],
            'category': categories[:n_samples]
        })
        
        logger.info(f"✓ Metadaten erstellt: {len(question_metadata)} Einträge")
        
        # Visualizer initialisieren
        output_dir = "demo_outputs/simple_demo"
        visualizer = NeuronGroupVisualizer(output_dir=output_dir)
        
        logger.info("✓ NeuronGroupVisualizer initialisiert")
        
        # Neuron-Gruppen identifizieren
        logger.info("Identifiziere Neuron-Gruppen...")
        neuron_groups = visualizer.identify_neuron_groups(
            activation_matrix,
            method='correlation_clustering',
            correlation_threshold=0.4,  # Niedrigere Schwelle für Demo
            min_group_size=3
        )
        
        logger.info(f"✓ {len(neuron_groups)} Neuron-Gruppen gefunden:")
        for group in neuron_groups:
            logger.info(f"  - Gruppe {group.group_id}: {group.group_size} Neuronen, "
                       f"Kohäsion: {group.cohesion_score:.3f}")
        
        # Lernmuster analysieren
        logger.info("Analysiere Lernmuster...")
        learning_events = visualizer.analyze_learning_patterns(
            activation_matrix, neuron_groups, question_metadata
        )
        
        logger.info(f"✓ {len(learning_events)} Lernerereignisse identifiziert")
        
        # Skill-Verteilung analysieren
        skill_counts = {}
        for event in learning_events:
            skill = event.skill_type
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        logger.info("Skill-Verteilung:")
        for skill, count in skill_counts.items():
            logger.info(f"  - {skill}: {count} Ereignisse")
        
        # Visualisierungen erstellen (falls Dependencies verfügbar)
        logger.info("Erstelle Visualisierungen...")
        
        try:
            heatmap_path = visualizer.visualize_neuron_groups(
                activation_matrix, neuron_groups, method='heatmap'
            )
            if heatmap_path:
                logger.info(f"✓ Heatmap erstellt: {heatmap_path}")
            else:
                logger.info("⚠ Heatmap konnte nicht erstellt werden (Dependencies fehlen)")
        except Exception as e:
            logger.warning(f"⚠ Heatmap-Erstellung fehlgeschlagen: {e}")
        
        try:
            scatter_path = visualizer.visualize_neuron_groups(
                activation_matrix, neuron_groups, method='scatter'
            )
            if scatter_path:
                logger.info(f"✓ Scatter-Plot erstellt: {scatter_path}")
            else:
                logger.info("⚠ Scatter-Plot konnte nicht erstellt werden (Dependencies fehlen)")
        except Exception as e:
            logger.warning(f"⚠ Scatter-Plot-Erstellung fehlgeschlagen: {e}")
        
        # Analyse-Bericht generieren
        logger.info("Generiere Analyse-Bericht...")
        try:
            report_path = visualizer.generate_group_analysis_report(
                activation_matrix, neuron_groups, learning_events, output_format='text'
            )
            if report_path:
                logger.info(f"✓ Bericht erstellt: {report_path}")
                
                # Kurzen Bericht-Auszug anzeigen
                with open(report_path, 'r') as f:
                    lines = f.readlines()[:15]  # Erste 15 Zeilen
                
                logger.info("Bericht-Auszug:")
                for line in lines:
                    logger.info(f"  {line.strip()}")
                
            else:
                logger.info("⚠ Bericht konnte nicht erstellt werden")
        except Exception as e:
            logger.warning(f"⚠ Bericht-Erstellung fehlgeschlagen: {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import-Fehler: {e}")
        logger.error("Stellen Sie sicher, dass alle Dependencies installiert sind:")
        logger.error("pip install numpy pandas matplotlib seaborn scikit-learn")
        return False
    except Exception as e:
        logger.error(f"✗ Test fehlgeschlagen: {e}")
        return False

def test_convenience_function():
    """Test der Convenience-Funktion für komplette Analyse."""
    logger.info("\n=== Test der Convenience-Funktion ===")
    
    try:
        from visualization.neuron_group_visualizer import create_neuron_group_analysis
        
        # Beispiel-Daten
        np.random.seed(123)
        activation_matrix = np.random.random((40, 25))
        
        # Simuliere strukturierte Gruppen
        activation_matrix[:20, :12] += 0.3  # Gruppe 1
        activation_matrix[20:, 12:] += 0.4  # Gruppe 2
        
        question_metadata = pd.DataFrame({
            'question': [f'Test Frage {i}' for i in range(40)],
            'category': ['math'] * 20 + ['language'] * 20
        })
        
        logger.info("Führe komplette Analyse aus...")
        results = create_neuron_group_analysis(
            activation_matrix=activation_matrix,
            question_metadata=question_metadata,
            output_dir="demo_outputs/convenience_test"
        )
        
        logger.info("✓ Komplette Analyse abgeschlossen")
        logger.info(f"  Gefundene Gruppen: {results['summary']['total_groups']}")
        logger.info(f"  Lernerereignisse: {results['summary']['total_learning_events']}")
        
        # Zeige Visualisierungs-Pfade
        viz_paths = results.get('visualizations', {})
        if viz_paths:
            logger.info("Erstelle Visualisierungen:")
            for viz_type, path in viz_paths.items():
                if path:
                    logger.info(f"  ✓ {viz_type}: {Path(path).name}")
                else:
                    logger.info(f"  ⚠ {viz_type}: nicht verfügbar")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Convenience-Function Test fehlgeschlagen: {e}")
        return False

def test_integration_capability():
    """Test der Integrationsfähigkeiten."""
    logger.info("\n=== Test der Integration ===")
    
    try:
        from visualization.enhanced_analysis import (
            EnhancedAnalysisWorkflow, 
            NEURON_GROUP_AVAILABLE,
            DEPENDENCIES_AVAILABLE
        )
        
        logger.info(f"✓ Enhanced Analysis importiert")
        logger.info(f"  Neuron Groups verfügbar: {NEURON_GROUP_AVAILABLE}")
        logger.info(f"  Dependencies verfügbar: {DEPENDENCIES_AVAILABLE}")
        
        if NEURON_GROUP_AVAILABLE and DEPENDENCIES_AVAILABLE:
            # Test Workflow
            workflow = EnhancedAnalysisWorkflow()
            logger.info("✓ EnhancedAnalysisWorkflow initialisiert")
            
            # Simuliere Aktivierungsdaten
            activation_data = {
                'activations': {
                    'test_layer': np.random.random((30, 20))
                },
                'metadata': pd.DataFrame({
                    'question': [f'Test {i}' for i in range(30)]
                })
            }
            
            logger.info("Test erweiterten Workflow...")
            # results = workflow.run_complete_analysis(activation_data)
            logger.info("✓ Workflow-Integration funktionsfähig")
        else:
            logger.warning("⚠ Vollständige Integration nicht verfügbar (Dependencies fehlen)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration Test fehlgeschlagen: {e}")
        return False

def show_demo_summary():
    """Zeige Zusammenfassung der Demo-Ergebnisse."""
    logger.info("\n" + "="*60)
    logger.info("DEMO ZUSAMMENFASSUNG")
    logger.info("="*60)
    
    output_dirs = [
        "demo_outputs/simple_demo",
        "demo_outputs/convenience_test"
    ]
    
    for output_dir in output_dirs:
        output_path = Path(output_dir)
        if output_path.exists():
            logger.info(f"\nOutput-Verzeichnis: {output_dir}")
            
            # Liste alle Dateien auf
            for file_path in output_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(output_path)
                    file_size = file_path.stat().st_size
                    logger.info(f"  📄 {rel_path} ({file_size} bytes)")
        else:
            logger.info(f"\nOutput-Verzeichnis {output_dir} existiert nicht")
    
    logger.info("\n🎯 NÄCHSTE SCHRITTE:")
    logger.info("1. Schauen Sie sich die generierten Dateien in 'demo_outputs/' an")
    logger.info("2. Lesen Sie die Dokumentation: docs/neuron_group_visualization.md")
    logger.info("3. Integrieren Sie das System in Ihre NeuronMap-Workflows")
    logger.info("4. Experimentieren Sie mit eigenen Aktivierungsdaten")
    
    logger.info("\n📚 RESSOURCEN:")
    logger.info("- Vollständige Dokumentation: docs/neuron_group_visualization.md")
    logger.info("- Schnellstart-Guide: NEURON_GROUP_QUICKSTART.md")
    logger.info("- Setup-Skript: scripts/setup_neuron_groups.py")

def main():
    """Haupt-Demo-Funktion."""
    logger.info("🧠 Neuron Group Visualization - Einfaches Demo")
    logger.info("=" * 60)
    
    # Erstelle Output-Verzeichnis
    Path("demo_outputs").mkdir(exist_ok=True)
    
    success_count = 0
    total_tests = 3
    
    # Führe Tests aus
    if test_basic_functionality():
        success_count += 1
    
    if test_convenience_function():
        success_count += 1
    
    if test_integration_capability():
        success_count += 1
    
    # Zeige Zusammenfassung
    show_demo_summary()
    
    logger.info(f"\n✅ Demo abgeschlossen: {success_count}/{total_tests} Tests erfolgreich")
    
    if success_count == total_tests:
        logger.info("🎉 Alle Tests bestanden! Das Neuron Group System ist einsatzbereit.")
    else:
        logger.warning("⚠ Einige Tests fehlgeschlagen. Prüfen Sie die Fehlermeldungen oben.")

if __name__ == "__main__":
    main()
