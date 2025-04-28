# Block 1: Imports und Konfiguration
# ====================================
import pandas as pd
import numpy as np
import ast  # Zum sicheren Auswerten von String-Literalen (für die Vektor-Listen)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler # Optional: Für die Normalisierung der Daten
import sys # Für saubere Fehlermeldungen

# --- Konfiguration ---
INPUT_CSV = "activation_results.csv" # Die vom ersten Skript erzeugte Datei
OUTPUT_PLOT_PCA = "activation_pca_scatter.png"
OUTPUT_PLOT_TSNE = "activation_tsne_scatter.png"
OUTPUT_PLOT_HEATMAP = "activation_heatmap.png" # Optional

# Parameter für Dimensionsreduktion
N_COMPONENTS = 2 # Wir wollen auf 2 Dimensionen für den Scatterplot reduzieren
TSNE_PERPLEXITY = 30 # Typischer Wert für t-SNE, kann angepasst werden (5-50)
TSNE_LEARNING_RATE = 200 # Typischer Wert für t-SNE
TSNE_N_ITER = 1000 # Anzahl Iterationen für t-SNE

# Parameter für die Heatmap (optional)
HEATMAP_MAX_QUESTIONS = 50 # Zeige nur die ersten X Fragen in der Heatmap (sonst unlesbar)
HEATMAP_MAX_NEURONS = 100 # Zeige nur die ersten Y Neuronen/Dimensionen (sonst unlesbar)
# --------------------


# Block 2: Daten laden und vorbereiten
# ===================================

def load_and_prepare_data(filepath):
    """Lädt die CSV, parst die Vektoren und bereitet die Matrix vor."""
    try:
        df = pd.read_csv(filepath)
        print(f"INFO: Daten aus '{filepath}' geladen. Anzahl Zeilen: {len(df)}")
    except FileNotFoundError:
        print(f"FEHLER: Die Datei '{filepath}' wurde nicht gefunden. Stelle sicher, dass das erste Skript erfolgreich lief.")
        sys.exit(1)
    except Exception as e:
        print(f"FEHLER: Ein unerwarteter Fehler trat beim Laden der CSV auf: {e}")
        sys.exit(1)

    # Entferne Zeilen, bei denen der Aktivierungsvektor fehlt (falls im ersten Skript Fehler auftraten)
    original_len = len(df)
    df.dropna(subset=['activation_vector'], inplace=True)
    if len(df) < original_len:
        print(f"WARNUNG: {original_len - len(df)} Zeilen ohne Aktivierungsvektor entfernt.")

    if df.empty:
        print("FEHLER: Keine gültigen Aktivierungsvektoren in der Datei gefunden.")
        sys.exit(1)

    # Die 'activation_vector'-Spalte enthält Strings wie "[0.1, 0.2, ...]".
    # Wir müssen sie sicher in echte Listen von Zahlen umwandeln.
    try:
        # ast.literal_eval ist sicherer als eval()
        df['activation_list'] = df['activation_vector'].apply(ast.literal_eval)
    except (ValueError, SyntaxError) as e:
         print(f"FEHLER: Konnte die 'activation_vector'-Spalte nicht korrekt als Liste parsen.")
         print("Stelle sicher, dass die Vektoren im Format '[num, num, ...]' gespeichert sind.")
         print(f"Fehlerdetails: {e}")
         # Zeige die ersten paar fehlerhaften Einträge
         print("\nBeispielhafte Einträge aus 'activation_vector':")
         for i, vec_str in enumerate(df['activation_vector'].head()):
            print(f"Zeile {i}: {vec_str[:100]}...") # Zeige nur Anfang des Strings
         sys.exit(1)


    # Überprüfe, ob alle Vektoren die gleiche Länge haben
    vector_lengths = df['activation_list'].apply(len)
    if vector_lengths.nunique() > 1:
        print("WARNUNG: Die Aktivierungsvektoren haben unterschiedliche Längen!")
        print(f"Gefundene Längen: {vector_lengths.unique()}")
        # Optional: Hier könnte man entscheiden, wie man damit umgeht (z.B. auf min/max Länge padden/kürzen oder Fehler werfen)
        # Fürs Erste machen wir weiter, aber PCA/t-SNE erwarten gleiche Längen. Wir nehmen die erste Länge als Referenz.
        target_len = vector_lengths.iloc[0]
        print(f"INFO: Versuche, mit der ersten gefundenen Länge ({target_len}) weiterzuarbeiten. Filterung könnte nötig sein.")
        df = df[df['activation_list'].apply(len) == target_len].copy()
        if df.empty:
            print("FEHLER: Nach Filterung auf einheitliche Vektorlänge sind keine Daten mehr übrig.")
            sys.exit(1)

    # Konvertiere die Listen von Vektoren in eine NumPy-Matrix
    # Jede Zeile der Matrix ist ein Aktivierungsvektor
    activation_matrix = np.array(df['activation_list'].tolist())
    
    print(f"INFO: Aktivierungsmatrix erstellt mit Shape: {activation_matrix.shape}") # (Anzahl Fragen, Anzahl Neuronen/Dimensionen)

    # Optional, aber oft empfohlen: Daten normalisieren (Standardisierung)
    # Das hilft Algorithmen wie PCA und t-SNE, besser zu funktionieren.
    scaler = StandardScaler()
    activation_matrix_scaled = scaler.fit_transform(activation_matrix)
    print("INFO: Aktivierungsmatrix normalisiert (StandardScaler).")

    return df, activation_matrix_scaled # Gib DataFrame und die (skalierte) Matrix zurück

# Block 3: Dimensionsreduktion
# ============================

def run_pca(data_matrix, n_components=2):
    """Führt PCA auf der Datenmatrix aus."""
    print(f"INFO: Führe PCA mit {n_components} Komponenten durch...")
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_matrix)
    print(f"INFO: PCA abgeschlossen. Erklärte Varianz durch {n_components} Komponenten: {np.sum(pca.explained_variance_ratio_):.4f}")
    return pca_result

def run_tsne(data_matrix, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
    """Führt t-SNE auf der Datenmatrix aus."""
    print(f"INFO: Führe t-SNE mit {n_components} Komponenten durch (dies kann dauern)...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42) # random_state für Reproduzierbarkeit
    tsne_result = tsne.fit_transform(data_matrix)
    print("INFO: t-SNE abgeschlossen.")
    return tsne_result

# Block 4: Visualisierung
# =======================

def plot_scatter(data_2d, title, filename, df_questions=None):
    """Erstellt einen 2D Scatterplot und speichert ihn."""
    plt.figure(figsize=(12, 10))
    
    # Erstelle den Scatterplot
    scatter = sns.scatterplot(
        x=data_2d[:, 0], 
        y=data_2d[:, 1],
        s=50,  # Punktgröße
        alpha=0.7 # Transparenz
        # Optional: hue=df_questions['cluster_label'] # Wenn Clustering durchgeführt wurde
    )
    
    plt.title(title, fontsize=16)
    plt.xlabel("Komponente 1", fontsize=12)
    plt.ylabel("Komponente 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Optional: Tooltips oder Beschriftungen hinzufügen (kann bei 500 Punkten unübersichtlich werden)
    # for i, txt in enumerate(df_questions['question'].head(20)): # Nur die ersten 20 beschriften
    #    plt.annotate(f"Q{i}", (data_2d[i, 0], data_2d[i, 1]))
            
    plt.tight_layout() # Sorgt für gute Abstände
    plt.savefig(filename, dpi=300) # Speichert den Plot in hoher Auflösung
    print(f"INFO: Scatterplot gespeichert als '{filename}'")
    plt.close() # Schließt die Figur, um Speicher freizugeben

def plot_heatmap(activation_matrix, df_questions, filename, max_questions, max_neurons):
    """Erstellt eine Heatmap der Aktivierungen (Ausschnitt) und speichert sie."""
    if activation_matrix.shape[0] == 0 or activation_matrix.shape[1] == 0:
        print("WARNUNG: Kann keine Heatmap erstellen, da die Matrix leer ist.")
        return
        
    # Reduziere die Matrix auf eine handhabbare Größe für die Visualisierung
    num_questions = min(max_questions, activation_matrix.shape[0])
    num_neurons = min(max_neurons, activation_matrix.shape[1])
    heatmap_data = activation_matrix[:num_questions, :num_neurons]
    
    # Erstelle Frage-Labels (optional, kann bei langen Fragen zu viel Platz brauchen)
    # question_labels = [f"Q{i}: {q[:30]}..." for i, q in enumerate(df_questions['question'][:num_questions])]
    question_labels = [f"Frage {i}" for i in range(num_questions)] # Kürzere Labels
    neuron_labels = [f"Neuron {j}" for j in range(num_neurons)]
    
    plt.figure(figsize=(15, 10)) # Passe die Größe nach Bedarf an
    sns.heatmap(heatmap_data, 
                cmap="viridis", # Wähle eine Farbpalette (z.B. "viridis", "coolwarm", "magma")
                xticklabels=neuron_labels, 
                yticklabels=question_labels)
    plt.title(f"Heatmap der Aktivierungen (Erste {num_questions} Fragen vs. Erste {num_neurons} Neuronen)", fontsize=14)
    plt.xlabel("Neuron / Dimension im Aktivierungsvektor", fontsize=10)
    plt.ylabel("Frage Index", fontsize=10)
    plt.xticks(rotation=90, fontsize=8) # Rotiere X-Achsen-Beschriftung
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"INFO: Heatmap gespeichert als '{filename}'")
    plt.close()

# Block 5: Hauptausführung
# ========================

def main():
    # 1. Daten laden und vorbereiten
    df_questions, activation_matrix = load_and_prepare_data(INPUT_CSV)

    # --- Visualisierung 1: PCA ---
    pca_result_2d = run_pca(activation_matrix, n_components=N_COMPONENTS)
    plot_scatter(pca_result_2d, 
                 f"PCA der Neuronenaktivierungen ({N_COMPONENTS} Komponenten)", 
                 OUTPUT_PLOT_PCA, 
                 df_questions)

    # --- Visualisierung 2: t-SNE ---
    # Hinweis: t-SNE kann bei vielen Datenpunkten (>1000) langsam sein.
    # Bei 500 Punkten sollte es aber gut funktionieren.
    tsne_result_2d = run_tsne(activation_matrix, 
                              n_components=N_COMPONENTS, 
                              perplexity=TSNE_PERPLEXITY, 
                              learning_rate=TSNE_LEARNING_RATE, 
                              n_iter=TSNE_N_ITER)
    plot_scatter(tsne_result_2d, 
                 f"t-SNE der Neuronenaktivierungen ({N_COMPONENTS} Komponenten, Perplexity={TSNE_PERPLEXITY})", 
                 OUTPUT_PLOT_TSNE, 
                 df_questions)

    # --- Visualisierung 3: Heatmap (Optional) ---
    # Nutze hier die *skalierte* Matrix, um Farbunterschiede besser sichtbar zu machen
    plot_heatmap(activation_matrix, # oder nimm activation_matrix_raw für Originalwerte
                 df_questions, 
                 OUTPUT_PLOT_HEATMAP, 
                 HEATMAP_MAX_QUESTIONS, 
                 HEATMAP_MAX_NEURONS)

    print("\nINFO: Visualisierungs-Skript erfolgreich abgeschlossen.")

if __name__ == "__main__":
    main()
