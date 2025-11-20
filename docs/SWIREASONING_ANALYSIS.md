# SwiReasoning: Analyse & Architektur-Konzept

## 1. Zusammenfassung des Papers (SwiReasoning)
**Titel:** SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs
**Kernidee:** Ein hybrider Ansatz, der die Vorteile von explizitem "Chain-of-Thought" (CoT) Reasoning und latentem (internem) Reasoning kombiniert.

### Schlüsselmechanismen
1.  **Dualer Modus:**
    *   **Explizit:** Das Modell generiert Tokens (Gedankenschritte) in natürlicher Sprache.
    *   **Latent:** Das Modell verarbeitet Informationen in internen Hidden States ohne Token-Output (oder mit speziellen "Thinking Tokens").
2.  **Dynamisches Switching:**
    *   Das System entscheidet zur Laufzeit, wann zwischen den Modi gewechselt wird.
    *   **Trigger:** Basierend auf der Entropie der Next-Token-Distribution. Hohe Entropie (Unsicherheit) -> Switch zu explizitem Reasoning? Oder umgekehrt? (Laut Paper: Entropie-Trends als Konfidenz-Maß).
3.  **Effizienz:**
    *   Reduziert die Token-Anzahl drastisch, da einfache Schritte latent verarbeitet werden.
    *   Erhöht die Genauigkeit bei komplexen Aufgaben durch explizite Schritte.

## 2. Integration in NeuronMap (Guardian Framework)

Wir werden SwiReasoning nicht als separates Modell, sondern als **Policy** innerhalb des Guardian-Frameworks implementieren.

### Architektur-Mapping

| SwiReasoning Komponente | NeuronMap / Guardian Äquivalent |
| :--- | :--- |
| **Confidence Estimation** | `src/guardian/probes.py` (Entropy Probe) |
| **Switching Logic** | `src/guardian/policies.py` (`SwiReasoningPolicy`) |
| **Latent Reasoning** | `src/guardian/interventions.py` (Unterdrückung von Output, interne Rechenschritte) |
| **Explicit Reasoning** | Standard Generation (Normaler Forward Pass) |

### 3. Implementierungs-Strategie

#### Schritt 1: Erweiterte Probes
Wir benötigen präzisere Metriken als nur die einfache Entropie.
*   **Block-wise Entropy:** Durchschnittliche Entropie über ein Fenster von Tokens.
*   **Entropy Trend:** Steigt oder fällt die Unsicherheit?

#### Schritt 2: Die SwiReasoning Policy
Eine neue Klasse `SwiReasoningPolicy(GuardianPolicy)`:
*   **Input:** Aktuelle Entropie, Historie der Entropie.
*   **State:** `LATENT` oder `EXPLICIT`.
*   **Action:**
    *   Wenn `LATENT` und Unsicherheit steigt -> Switch zu `EXPLICIT` (Output generieren).
    *   Wenn `EXPLICIT` und Konfidenz hoch -> Switch zu `LATENT` (Output unterdrücken / interne Tokens nutzen).

#### Schritt 3: Visualisierung
Der `RealtimeStreamer` muss anzeigen, in welchem Modus sich das Modell befindet.
*   **Farzkodierung:** Blau für Latent, Orange für Explizit.
*   **Graph:** Entropie-Verlauf in Echtzeit.

## 4. Offene Fragen & Risiken
*   **Latent Reasoning Simulation:** Da wir mit fertigen Modellen (z.B. GPT-2/Llama) arbeiten, können wir "Latent Reasoning" nicht ohne Weiteres erzwingen, wenn das Modell nicht dafür trainiert wurde (z.B. mit `<thinking>` Tokens).
*   **Workaround:** Wir simulieren den Effekt, indem wir das Modell "leise" weiterrechnen lassen (interne Forward Passes ohne Output) oder indem wir spezielle "Reasoning Tokens" injizieren, die im Output ausgeblendet werden.

## 5. Nächste Schritte
1.  Implementierung der Datenstrukturen (`ThinkingBlock`, `ReasoningSwitch`).
2.  Erweiterung der `GuardianEngine` um den `SwiReasoning`-Modus.
