# Mathematische Formeln in NeuronMap

Dieses Dokument enthält eine Sammlung der wichtigsten mathematischen Formeln, die im NeuronMap-Intervention-System verwendet werden. Die Formeln sind präzise dargestellt und stammen hauptsächlich aus `src/analysis/interventions.py` und den fundamentalen Architekturen, die das System analysiert.

## 1. Formeln für Interventions-Typen

Sei $A$ der Aktivierungsvektor eines Neurons oder Layers und $A'$ der modifizierte Aktivierungsvektor.

### Ablation
Das Setzen von Aktivierungen auf Null:
$$ A'_{\text{target}} = 0 $$
Für alle Zielneuronen (${\text{target}}$) wird die Aktivierung auf Null gesetzt. Wenn keine spezifischen Neuronen angegeben sind, werden alle Aktivierungen des Layers auf Null gesetzt.

### Rauschen (Noise)
Das Hinzufügen von stochastischem Rauschen zu den Aktivierungen:
$$ A' = A + \epsilon $$
wobei $\epsilon \sim \mathcal{N}(0, \sigma^2)$ ein Rauschvektor ist, der aus einer Normalverteilung mit Mittelwert 0 und Standardabweichung $\sigma$ (hier `intervention_value` oder 0.1) gezogen wird und die gleiche Form wie $A$ aufweist.

### Mittelwert (Mean)
Das Ersetzen von Aktivierungen durch den Mittelwert:
$$ A'_{\text{target}} = \text{mean}(A) $$
Für alle Zielneuronen (${\text{target}}$) wird die Aktivierung durch den mittleren Wert aller Aktivierungen im Tensor $A$ ersetzt. Wenn keine spezifischen Neuronen angegeben sind, wird der gesamte Tensor mit seinem Mittelwert gefüllt.

### Patching
Das Ersetzen von Aktivierungen aus einer "korrupten" Ausführung durch Aktivierungen aus einer "sauberen" Ausführung:
$$ A'_{\text{corrupted, target}} = A_{\text{clean, target}} $$
Hierbei werden die Aktivierungen von Zielneuronen (${\text{target}}$) im ursprünglich "korrupten" Pfad durch die entsprechenden Aktivierungen aus dem "sauberen" Pfad ersetzt.

## 2. Formeln zur Berechnung des kausalen Effekts

Die Funktion `calculate_causal_effect` quantifiziert den kausalen Einfluss anhand des Vergleichs von Ausgaben. Sei $O_{\text{clean}}$, $O_{\text{corrupted}}$ und $O_{\text{patched}}$ die Ausgaben des Modells für die "saubere" Eingabe, die "korrumpierte" Eingabe (ohne Patching) und die "gepatchte" Eingabe (korrumpiert mit überlagerten sauberen Aktivierungen).

### Metrik: Logit-Differenz (`logit_diff`)
Diese Metrik verwendet die euklidische Norm (L2-Norm) der Logit-Vektoren:
$$ \text{Kausaler Effekt} = \frac{\|O_{\text{patched}} - O_{\text{corrupted}}\|_2}{\|O_{\text{clean}} - O_{\text{corrupted}}\|_2 + \epsilon} $$

- $\|\cdot\|_2$: Euklidische Norm (L2-Norm) des Vektors.
- $\epsilon$: Ein kleiner Wert (z.B. $10^{-8}$) zur Vermeidung von Division durch Null.

**Interpretation:** Misst, wie stark das Patchen die Ausgabe der korrumpierten Version in Richtung der sauberen Version verschiebt.

### Metrik: Wahrscheinlichkeits-Differenz (`probability`)
Diese Metrik vergleicht die Wahrscheinlichkeiten des Top-Tokens ($t^*$) der sauberen Ausführung:
$$ \text{Kausaler Effekt} = \frac{P(t^*|O_{\text{patched}}) - P(t^*|O_{\text{corrupted}})}{P(t^*|O_{\text{clean}}) - P(t^*|O_{\text{corrupted}}) + \epsilon} $$

- $P(t^*|O)$: Die Wahrscheinlichkeit des Top-Tokens $t^*$ (ermittelt aus $O_{\text{clean}}$) in der jeweiligen Modellausgabe $O$.

**Interpretation:** Misst, wie gut das Patchen die Wahrscheinlichkeit des ursprünglichen Top-Tokens der sauberen Ausführung wiederherstellt.

---

## 3. Formeln der Transformer-Architektur

Da das System primär zur Analyse von Transformer-Modellen (wie GPT, BERT, Llama) dient, ist die Mathematik des **Self-Attention-Mechanismus** von zentraler Bedeutung. Dies ist die Kernformel, die in den analysierten Modellen steckt.

### Scaled Dot-Product Attention

Die Attention wird wie folgt berechnet:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

- **Input:** Die Eingabe für einen Self-Attention-Layer besteht aus einer Sequenz von Vektoren. Aus jedem dieser Vektoren werden drei neue Vektoren erzeugt:
  - $Q$ (Query): Eine Matrix, die alle "Anfrage"-Vektoren der Sequenz enthält. Sie repräsentiert, wonach ein Wort sucht.
  - $K$ (Key): Eine Matrix mit allen "Schlüssel"-Vektoren. Sie repräsentiert, was ein Wort an Information anbietet.
  - $V$ (Value): Eine Matrix mit allen "Wert"-Vektoren. Sie repräsentiert die eigentliche Information, die ein Wort trägt.

- **Berechnungsschritte:**
  1.  **$QK^T$**: Berechnet die Ähnlichkeit (Skalarprodukt) zwischen jedem Query-Vektor und jedem Key-Vektor. Das Ergebnis ist eine Aufmerksamkeits-Matrix, die zeigt, wie stark jedes Wort auf jedes andere Wort in der Sequenz achten sollte.
  2.  **$\frac{...}{\sqrt{d_k}}$**: Skaliert die Ähnlichkeitswerte, um stabile Gradienten während des Trainings zu gewährleisten. $d_k$ ist die Dimension der Key-Vektoren.
  3.  **$\text{softmax}(...)$**: Wandelt die skalierten Ähnlichkeitswerte in Wahrscheinlichkeiten (zwischen 0 und 1) um. Die Summe der Aufmerksamkeits-Werte für jedes Wort über die gesamte Sequenz ist 1.
  4.  **$...V$**: Multipliziert die resultierenden Aufmerksamkeits-Wahrscheinlichkeiten mit den Value-Vektoren. Dadurch werden die Informationen der Wörter, auf die am meisten geachtet werden soll, stärker gewichtet, während unwichtige Wörter unterdrückt werden.

- **Output:** Das Ergebnis ist eine neue Sequenz von Vektoren, bei der jeder Vektor eine kontextualisierte Repräsentation des ursprünglichen Wortes ist, angereichert mit Informationen aus der gesamten Sequenz.

### Multi-Head Attention

In der Praxis wird nicht nur eine einzige Attention-Berechnung durchgeführt, sondern mehrere parallel ("Multi-Head"). Jede "Head" lernt unterschiedliche Aspekte der Beziehungen zwischen Wörtern. Die Ergebnisse der einzelnen Heads werden am Ende konkateniert und linear transformiert, um die endgültige Ausgabe zu erzeugen.
