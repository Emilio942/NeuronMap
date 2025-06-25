# Block 1: Imports und grundlegende Konfiguration
# ==================================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm  # Für eine saubere Fortschrittsanzeige im Terminal
import json # Falls die Fragen als JSON pro Zeile gespeichert sind
import sys # Für saubere Fehlermeldungen

# --- Globale Konfiguration ---
# Wähle ein kleines Modell (z.B. "distilgpt2", "gpt2", oder spezifischere wie "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MODEL_NAME = "distilgpt2" 
# Pfad zur Datei mit den Fragen (eine Frage pro Zeile, oder JSON pro Zeile - siehe load_questions)
QUESTIONS_FILE = "generated_questions.jsonl" 
# ZielsCHICHT: Dies ist ein BEISPIEL! Muss evtl. angepasst werden.
# Finde den genauen Namen durch Untersuchen des Modells (siehe Funktion print_model_layers)
# Übliche Namen enthalten '.h[layer_index].' gefolgt von 'mlp', 'attn', 'c_proj', 'fc1', 'fc2' etc.
# Beispiel für distilgpt2: 'transformer.h.5.mlp.c_proj' (Ausgang des MLP in der letzten Schicht)
# Beispiel für distilgpt2: 'transformer.h.5.attn.c_attn' (Ausgang der Attention in der letzten Schicht)
TARGET_LAYER_NAME = "transformer.h.5.mlp.c_proj" # <-- ANPASSEN!
# Name der Ausgabedatei für die Ergebnisse
OUTPUT_FILE = "activation_results.csv"
# Verwende GPU, falls verfügbar, sonst CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"INFO: Using device: {DEVICE}")

# Speicher für die Aktivierung, die vom Hook erfasst wird
activation_capture = {} 
# -----------------------------


# Block 2: Hilfsfunktionen
# =========================

def load_questions(filepath):
    """Lädt Fragen aus einer JSON Lines Datei (.jsonl).
    
    Annahme: Jede Zeile ist ein JSON-Objekt mit einem Schlüssel 'question'.
    """
    questions = [] # Initialisiere leere Liste
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f): # Nummeriere Zeilen für bessere Fehlermeldungen
                if line.strip(): # Überspringe leere Zeilen
                    try:
                        # --- AKTIVIERE DIESEN TEIL ---
                        data = json.loads(line)
                        if 'question' in data:
                            questions.append(data['question'])
                        else:
                            print(f"WARNUNG: Zeile {line_num+1}: JSON-Objekt hat keinen Schlüssel 'question'. Inhalt: {line.strip()}")
                        # ------------------------------
                    except json.JSONDecodeError as e:
                        print(f"WARNUNG: Konnte Zeile {line_num+1} nicht als JSON parsen: {line.strip()} - Fehler: {e}")

        # --- KOMMENTIERE DIESEN TEIL AUS ODER LÖSCHE IHN ---
        # Standard: Eine Frage pro Zeile (nicht mehr benötigt)
        # questions = [line.strip() for line in f if line.strip()] 
        # -------------------------------------------------
        
        if not questions:
             print(f"WARNUNG: Keine Fragen aus '{filepath}' extrahiert. Ist die Datei leer oder das Format/der Schlüsselname ('question') falsch?")
        else:
             print(f"INFO: Successfully loaded {len(questions)} questions from {filepath}")
        return questions
        
    except FileNotFoundError:
        print(f"FEHLER: Die Datei '{filepath}' wurde nicht gefunden.")
        sys.exit(1) # Beendet das Skript
    except Exception as e:
        print(f"FEHLER: Ein unerwarteter Fehler trat beim Laden der Fragen auf: {e}")
        sys.exit(1)

def load_model_and_tokenizer(model_name, device):
    """Lädt das vortrainierte Modell und den Tokenizer."""
    print(f"INFO: Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Setze Padding Token, falls nicht vorhanden (wichtig für Modelle wie GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("INFO: No padding token found. Using EOS token as padding token.")
        
    print(f"INFO: Loading model '{model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()  # Wichtig: Modell in den Evaluationsmodus setzen
    print(f"INFO: Model '{model_name}' loaded successfully on {device}.")
    return tokenizer, model

def print_model_layers(model):
    """Gibt alle Layer-Namen des Modells aus, um den TARGET_LAYER_NAME zu finden."""
    print("\n--- Verfügbare Layer-Namen im Modell ---")
    for name, module in model.named_modules():
        print(name)
    print("---------------------------------------\n")

def find_target_layer(model, layer_name):
    """Findet das spezifische Layer-Objekt im Modell anhand seines Namens."""
    for name, module in model.named_modules():
        if name == layer_name:
            print(f"INFO: Target layer '{layer_name}' found.")
            return module
    # Falls der Layer nicht gefunden wurde:
    print(f"FEHLER: Target layer '{layer_name}' wurde nicht im Modell gefunden!")
    print_model_layers(model) # Zeige verfügbare Layer an, um bei der Auswahl zu helfen
    sys.exit(1) # Beendet das Skript


# Block 3: Hook-Funktion und Aktivierungsextraktion
# ==================================================

def hook_fn(module, input_hook, output_hook):
    """
    Diese Funktion wird aufgerufen, wenn der Forward-Pass den Ziel-Layer erreicht.
    Sie speichert die (aggregierte) Aktivierung des Layers.
    """
    # output_hook kann ein Tensor oder ein Tupel sein. Oft ist der erste Eintrag der relevante Tensor.
    if isinstance(output_hook, torch.Tensor):
        output_tensor = output_hook
    elif isinstance(output_hook, tuple) and len(output_hook) > 0 and isinstance(output_hook[0], torch.Tensor):
        output_tensor = output_hook[0]
    else:
        print(f"WARNUNG: Unerwarteter Output-Typ vom Layer: {type(output_hook)}. Überspringe Hook.")
        return

    # Aktivierung vom Rechengraph lösen und auf CPU verschieben
    detached_output = output_tensor.detach().cpu()
    
    # Aggregation: Mittelwert über die Sequenzlänge nehmen.
    # Annahme: Output-Shape ist (batch_size=1, sequence_length, hidden_dim)
    # Nach [0] ist es (sequence_length, hidden_dim). Wir mitteln über dim=0.
    if detached_output.ndim >= 2:
         # Prüfen ob Batch Dimension vorhanden ist (sollte 1 sein, da wir einzeln verarbeiten)
        if detached_output.shape[0] == 1:
             aggregated_activation = detached_output[0].mean(dim=0)
        else:
            # Falls kein expliziter Batch da ist (selten, aber möglich)
             aggregated_activation = detached_output.mean(dim=0)
        # Speichern der aggregierten Aktivierung (als flacher Vektor)
        activation_capture['activation'] = aggregated_activation
    else:
        print(f"WARNUNG: Unerwartete Tensor-Dimensionen für Aggregation: {detached_output.shape}. Speichere unverändert.")
        activation_capture['activation'] = detached_output # Fallback

def get_activation_for_question(question, tokenizer, model, target_layer, device):
    """
    Verarbeitet eine einzelne Frage, registriert den Hook, führt den Forward-Pass aus
    und gibt die erfasste Aktivierung zurück.
    """
    global activation_capture # Zugriff auf den globalen Speicher
    activation_capture.clear() # Sicherstellen, dass der Speicher leer ist

    # Hook registrieren
    hook_handle = target_layer.register_forward_hook(hook_fn)

    try:
        # Frage tokenisieren und auf das richtige Gerät verschieben
        # Truncation hinzugefügt, falls Frage zu lang ist
        inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length, padding=False) 
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward-Pass ohne Gradientenberechnung (spart Speicher und Rechenzeit)
        with torch.no_grad():
            outputs = model(**inputs) # Wir brauchen den Output des Modells hier nicht direkt

    except Exception as e:
        print(f"FEHLER beim Verarbeiten der Frage: '{question[:50]}...' - Fehler: {e}")
        hook_handle.remove() # Hook immer entfernen!
        return None # Signalisiert einen Fehler
    finally:
        # Hook MUSS immer entfernt werden, sonst bleibt er für zukünftige Pässe aktiv!
        hook_handle.remove()

    # Erfasste Aktivierung abrufen (sollte jetzt im Dictionary sein)
    captured_activation = activation_capture.get('activation', None)

    if captured_activation is None:
        print(f"WARNUNG: Keine Aktivierung für Frage erfasst: '{question[:50]}...'")
        return None
    
    # Aktivierung als NumPy-Array zurückgeben (einfacher für Pandas)
    return captured_activation.numpy()


# Block 4: Hauptverarbeitungsschleife
# ===================================

def main():
    """Hauptfunktion zur Orchestrierung des Prozesses."""
    
    # 1. Daten laden
    questions = load_questions(QUESTIONS_FILE)
    if not questions: # Beenden, falls keine Fragen geladen wurden
        return 

    # 2. Modell und Tokenizer laden
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME, DEVICE)

    # 3. Ziel-Layer finden (wichtig: nach dem Laden des Modells!)
    target_layer = find_target_layer(model, TARGET_LAYER_NAME)
    if target_layer is None: # Beenden, falls Layer nicht gefunden
        return

    # 4. Aktivierungen für jede Frage sammeln
    results = []
    print(f"\nINFO: Starte Verarbeitung von {len(questions)} Fragen...")
    # tqdm sorgt für eine Fortschrittsanzeige, ohne das Terminal zu überladen
    for question in tqdm(questions, desc="Analysiere Fragen", unit="Frage"):
        activation_vector = get_activation_for_question(question, tokenizer, model, target_layer, DEVICE)
        
        if activation_vector is not None:
            # Ergebnis speichern (Frage und der dazugehörige Aktivierungsvektor)
            results.append({
                'question': question,
                'activation_vector': activation_vector 
                # Optional: Nur einen Teil der Aktivierung speichern, falls zu groß
                # 'activation_vector': activation_vector[:100] 
            })
        else:
            # Fallback, falls die Aktivierung nicht erfasst werden konnte
             results.append({
                'question': question,
                'activation_vector': None # Markieren, dass hier etwas schiefging
            })

    print(f"\nINFO: Verarbeitung abgeschlossen. {len(results)} Ergebnisse gesammelt.")

    # 5. Ergebnisse in eine CSV-Datei speichern
    if results:
        print(f"INFO: Speichere Ergebnisse in '{OUTPUT_FILE}'...")
        df = pd.DataFrame(results)
        
        # Konvertiere NumPy-Arrays in Listen für bessere CSV-Kompatibilität (optional, aber oft hilfreich)
        df['activation_vector'] = df['activation_vector'].apply(lambda x: x.tolist() if x is not None else None)
        
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"INFO: Ergebnisse erfolgreich in '{OUTPUT_FILE}' gespeichert.")
    else:
        print("WARNUNG: Keine Ergebnisse zum Speichern vorhanden.")

# Block 5: Skriptausführung
# ==========================
if __name__ == "__main__":
    main()

    # Optional: Drucke Layernamen, wenn das Skript nur dafür ausgeführt wird
    # if len(sys.argv) > 1 and sys.argv[1] == '--print-layers':
    #     print("INFO: Drucke nur die Layer-Namen des Modells...")
    #     _, model = load_model_and_tokenizer(MODEL_NAME, DEVICE)
    #     print_model_layers(model)
    # else:
    #     main() # Führe die Hauptanalyse aus
