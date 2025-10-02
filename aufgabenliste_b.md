Verstanden. Hier ist die detaillierte Aufgabenliste für den Feature-Block **"'Model Surgery' & Path-Analyse"**.

Wir brechen diesen komplexen Block in handhabbare technische Aufgaben auf, gegliedert nach den von dir gewünschten Abschnitten und versehen mit Schätzungen für Komplexität und Priorität.

---

### **Aufgabenliste: "Model Surgery" & Path-Analyse**

#### **1. Backend & Core-Engine**

Dies ist das Fundament. Hier erweitern wir die Engine, um nicht nur zu lesen, sondern aktiv in den Forward-Pass des Modells einzugreifen.

| Aufgabe | Beschreibung | Komplexität | Priorität |
| :--- | :--- | :--- | :--- |
| **B1: Modifizierbare Forward-Hooks** | Erweitere das bestehende Hook-System, sodass es nicht nur Aktivierungen liest, sondern auch modifizieren kann (z.B. durch Rückgabe eines neuen Tensors im Hook). | **Schwer** | **Hoch** |
| **B2: "Intervention-Cache"** | Implementiere eine Caching-Schicht, die während eines Model-Runs Aktivierungen von einem "sauberen" Lauf speichert, um sie in einem "korrumpierten" Lauf zu verwenden. | **Mittel** | **Hoch** |
| **B3: Core-Funktion für Ablation** | Erstelle eine Funktion `run_with_ablation(model, prompt, layer, neuron_indices)`, die ein Neuron oder einen Attention Head "ausschaltet" (auf null setzt) und die resultierende Ausgabe zurückgibt. | **Mittel** | **Hoch** |
| **B4: Core-Funktion für Path Patching** | Erstelle die Kernlogik `run_with_patching(model, clean_prompt, corrupted_prompt, patch_spec)`, die das Path-Patching-Experiment durchführt: "clean run", "corrupted run" und das Patchen der Aktivierungen. | **Schwer** | **Hoch** |
| **B5: Kausale Effekt-Analyse** | Entwickle eine Analysefunktion, die den "kausalen Effekt" misst, z.B. durch Vergleich der Logit-Differenz oder der Wahrscheinlichkeit des "korrekten" Tokens zwischen dem gepatchten und dem korrumpierten Lauf. | **Mittel** | **Mittel** |
| **B6: Konfigurations-Schema** | Definiere ein robustes Schema (z.B. in Pydantic) für Interventions-Konfigurationen (YAML), das alle Parameter für Ablation und Patching klar strukturiert. | **Einfach** | **Hoch** |

---

#### **2. CLI-Integration**

Macht die neuen Backend-Funktionen über die Kommandozeile zugänglich und nutzbar für Skripte und automatisierte Analysen.

| Aufgabe | Beschreibung | Komplexität | Priorität |
| :--- | :--- | :--- | :--- |
| **C1: CLI-Befehl `analyze:ablate`** | Erstelle einen neuen Befehl, mit dem Nutzer eine Ablations-Analyse direkt ausführen können. Z.B. `neuronmap analyze:ablate --model gpt2 --prompt "..." --layer 8 --neuron 1234`. | **Mittel** | **Hoch** |
| **C2: CLI-Befehl `analyze:patch`** | Erstelle einen Befehl, der eine Path-Patching-Analyse basierend auf einer Konfigurationsdatei durchführt. Z.B. `neuronmap analyze:patch --config patch_experiment.yml`. | **Mittel** | **Mittel** |
| **C3: Ausgabeformatierung** | Gestalte eine klare und informative Text-Ausgabe für die CLI-Befehle, die die Ergebnisse der Intervention (z.B. "Output vor/nach Ablation") übersichtlich darstellt. | **Einfach** | **Mittel** |

---

#### **3. Web-Interface (UI/UX)**

Dies ist der "Wow-Faktor": die direkte, visuelle Interaktion mit dem Modell. Baut auf einem funktionierenden Backend auf.

| Aufgabe | Beschreibung | Komplexität | Priorität |
| :--- | :--- | :--- | :--- |
| **W1: Backend-API für Interventionen** | Erstelle API-Endpunkte (z.B. `/api/interventions/ablate`), die das Web-Frontend aufrufen kann, um interaktiv Analysen anzustoßen. | **Mittel** | **Hoch** |
| **W2: Interaktive Visualisierungen** | Mache bestehende Visualisierungen (z.B. Heatmaps) klickbar, sodass bei Klick auf ein Neuron dessen ID und Layer an das UI übergeben werden. | **Mittel** | **Mittel** |
| **W3: "Intervention Panel" UI** | Entwirf eine UI-Komponente (z.B. eine Seitenleiste), die nach dem Klick auf ein Neuron erscheint. Sie zeigt Infos an und enthält Buttons wie "Dieses Neuron ausschalten". | **Mittel** | **Mittel** |
| **W4: Ergebnis-Visualisierung (Ablation)** | Stelle die Ergebnisse der Ablation direkt im UI dar. Z.B. eine "Diff"-Ansicht der Textausgabe und eine Mini-Grafik der geänderten Token-Wahrscheinlichkeiten. | **Mittel** | **Mittel** |
| **W5: Causal Tracing UI (Formular)** | Erstelle eine neue Seite oder ein Modal im Dashboard, auf der Nutzer die Parameter für ein Path-Patching-Experiment eingeben können (Clean/Corrupted Prompt etc.). | **Mittel** | **Niedrig** |
| **W6: Visualisierung des "Causal Path"** | (Anspruchsvoll) Entwickle eine Graphen-Visualisierung des Modells, in der der "kausale Pfad" – also die Kette von Interventionen, die das Ergebnis am stärksten beeinflusst hat – farblich hervorgehoben wird. | **Schwer** | **Niedrig** |

---

### **Zusammenfassung und nächste Schritte**

Diese Liste sollte eine solide Grundlage für die Planung und Entwicklung dieses Features bieten.

* **Startpunkt:** Die Aufgaben mit Priorität **Hoch** im **Backend (B1-B4, B6)** sind der kritische Pfad. Ohne sie kann weder das CLI noch das UI funktionieren.
* **Erster Meilenstein:** Ein funktionierender `analyze:ablate` CLI-Befehl (**C1**), der auf **B1** und **B3** aufbaut. Das wäre der erste sichtbare Erfolg.

Sobald dieser Block abgearbeitet ist oder parallel dazu Aufgaben mit niedrigerer Priorität frei werden, können wir die Aufgabenliste für den nächsten Feature-Block – z.B. **"Die Entdeckung von Circuits"** – erstellen.
































Perfekt. Dann tauchen wir jetzt in die faszinierende Welt der "Circuits" ein. Dieser Schritt ist anspruchsvoller, da wir von der Analyse einzelner Komponenten zur Analyse ihrer Beziehungen und ihres Zusammenspiels übergehen.

Hier ist die detaillierte Aufgabenliste für den Feature-Block **"'Die Entdeckung von Circuits': Vom Neuron zur Funktionseinheit"**.

---

### **Aufgabenliste: "Die Entdeckung von Circuits"**

#### **1. Backend & Core-Engine**

Das Herzstück dieses Blocks. Wir entwickeln die analytischen Methoden, um funktionale Verbindungen zwischen den Komponenten des Modells zu finden und zu bewerten.

| Aufgabe | Beschreibung | Komplexität | Priorität |
| :--- | :--- | :--- | :--- |
| **B1: Analyse der Attention Head Komposition** | Implementiere ein Modul zur Berechnung der Komposition von Attention Heads. Bestimme, welche Heads in späteren Layern die Ausgabe von Heads in früheren Layern "lesen", indem ihre `W_OV`-Matrizen multipliziert werden. | **Schwer** | **Hoch** |
| **B2: Analyse der Neuron-zu-Head Verbindung** | Entwickle eine Methode, um den Einfluss von MLP-Neuronen auf nachfolgende Attention Heads zu quantifizieren. Dies kann über Gradienten oder die Analyse der Gewichtsmatrizen geschehen. | **Schwer** | **Hoch** |
| **B3: Datenstruktur für Graphen/Circuits** | Definiere eine robuste, Graphen-basierte Datenstruktur (z.B. mit `networkx`), um Circuits abzubilden. Knoten sind Neuronen/Heads, Kanten sind gewichtete Verbindungen. Diese Struktur muss serialisierbar sein (z.B. nach JSON, GraphML). | **Mittel** | **Hoch** |
| **B4: "Scanner" für Induction Heads** | Erstelle ein spezifisches Analyse-Skript, das basierend auf den Aufmerksamkeitsmustern gezielt nach Induction Heads sucht (z.B. hohe Aufmerksamkeit auf das Token nach einer früheren Instanz des aktuellen Tokens). | **Mittel** | **Hoch** |
| **B5: "Scanner" für Copying/Saliency Heads** | Erstelle ein Skript, das nach Heads sucht, die primär Informationen von wichtigen Positionen (z.B. dem ersten Token) kopieren. | **Mittel** | **Mittel** |
| **B6: Circuit-Verifizierung mit Kausal-Tools** | Integriere die in Phase 1 entwickelten Kausal-Tools. Erstelle eine Funktion `verify_circuit(circuit, prompt)`, die automatisch ein Ablationsexperiment für alle Komponenten des gefundenen Circuits durchführt, um seine Funktion zu validieren. | **Mittel** | **Mittel** |

---

#### **2. CLI-Integration**

Wir machen die komplexen Analysen über die Kommandozeile zugänglich, mit einem starken Fokus auf maschinenlesbare Ausgaben für den Agenten-Workflow.

| Aufgabe | Beschreibung | Komplexität | Priorität |
| :--- | :--- | :--- | :--- |
| **C1: Hauptbefehl `analyze:circuits`** | Erstelle einen neuen Hauptbefehl in der CLI als Einstiegspunkt für alle Circuit-Analysen. | **Einfach** | **Hoch** |
| **C2: Unterbefehle für Scanner** | Füge Unterbefehle hinzu, um spezifische Suchen zu starten, z.B. `... find-induction-heads` oder `... find-copying-heads`. | **Mittel** | **Hoch** |
| **C3: Maschinenlesbare Graph-Ausgabe** | Implementiere eine `--output graphml` (oder `--output json-graph`) Option, die die gefundene Circuit-Struktur (aus Aufgabe B3) in ein standardisiertes, Graphen-basiertes Format exportiert. | **Mittel** | **Hoch** |
| **C4: CLI-Befehl für Circuit-Verifizierung** | Erstelle einen Befehl `... verify-circuit --circuit-file circuit.graphml --prompt "..."`, der die Verifizierungs-Logik aus B6 anstößt. | **Mittel** | **Mittel** |

---

#### **3. Web-Interface (UI/UX)**

Die Visualisierung von Circuits ist entscheidend für das menschliche Verständnis. Hier schaffen wir eine intuitive Darstellung der komplexen Beziehungen.

| Aufgabe | Beschreibung | Komplexität | Priorität |
| :--- | :--- | :--- | :--- |
| **W1: API-Endpunkt für Circuit-Daten** | Erstelle einen API-Endpunkt `/api/circuits/{circuit_type}`, der die vom Backend gefundenen Circuit-Daten im Graphen-Format (JSON) für das Frontend bereitstellt. | **Mittel** | **Hoch** |
| **W2: Graphen-Visualisierungs-Komponente** | Integriere eine Graphen-Visualisierungsbibliothek (z.B. Cytoscape.js, D3.js) in das Dashboard, um die Circuit-Struktur darzustellen. | **Schwer** | **Mittel** |
| **W3: Interaktiver Circuit-Explorer** | Mache den Graphen interaktiv: Klick auf einen Knoten (Head/Neuron) zeigt dessen Detail-Statistiken (aus den bereits existierenden Analysen). Hover über eine Kante zeigt die Verbindungsstärke. | **Mittel** | **Mittel** |
| **W4: Verknüpfung von Text und Graph** | (Anspruchsvoll) Visualisiere, wie ein Circuit einen Beispiel-Prompt verarbeitet. Wenn der Nutzer über ein Token im Text hovert, werden die aktiven Pfade im Graphen hervorgehoben und umgekehrt. | **Schwer** | **Niedrig** |

---

### **Zusammenfassung und nächste Schritte**

Dieser Feature-Block ist ein großer Sprung nach vorn.

* **Startpunkt:** Die Backend-Aufgaben **B1, B2, B3** sind das Fundament. Parallel kann ein erster Scanner wie **B4** entwickelt werden. Ein erster Meilenstein ist, wenn der CLI-Befehl `find-induction-heads` eine maschinenlesbare Graph-Datei (`C3`) ausspucken kann.
* **Synergie:** Beachte die Synergie zwischen diesem und dem letzten Block. Mit den Tools aus Schritt 1 (`Model Surgery`) können wir die Hypothesen, die wir hier mit den Scannern generieren, direkt im Anschluss testen und validieren (`B6`, `C4`). Das ist der Kern eines wissenschaftlichen, hypothesen-getriebenen Workflows.

Bereit für den nächsten Block, sobald du es bist!




















































Absolut. Dann zerlegen wir jetzt den dritten und letzten Block des Hauptziels "Tiefere Erkenntnisse". Dieser Schritt ist forschungsintensiv, aber das Potenzial für grundlegend neue Einblicke ist enorm.

Hier ist die detaillierte Aufgabenliste für den Feature-Block **"'Die Sprache der Neuronen verstehen': Polysemantizität & Abstraktion"**.

---

### **Aufgabenliste: "Die Sprache der Neuronen verstehen"**

#### **1. Backend & Core-Engine**

Hier implementieren wir die Kerntechnologien: eine Pipeline zum Trainieren von Sparse Autoencoders (SAEs) und die Analysemodule, um deren Ergebnisse sowie die Abstraktion im Modell zu interpretieren.

| Aufgabe | Beschreibung | Komplexität | Priorität |
| :--- | :--- | :--- | :--- |
| **B1: SAE-Trainingspipeline** | Implementiere eine vollständige Pipeline zum Trainieren eines SAEs auf den Aktivierungen eines bestimmten Transformer-Layers. Dies beinhaltet Daten-Streaming, Aktivierungs-Extraktion und den Trainings-Loop (Reconstruction Loss + L1 Sparsity Loss). | **Schwer** | **Hoch** |
| **B2: SAE-Feature-Extraktion** | Erstelle eine Funktion, die rohe Aktivierungen durch den Encoder eines trainierten SAEs leitet, um die hochdimensionalen, spärlichen "Feature-Aktivierungen" zu erhalten. Diese werden unsere neue Analyseeinheit. | **Mittel** | **Hoch** |
| **B3: Analyse der "Max Activating Examples"** | Entwickle ein Modul, das für jedes Feature eines trainierten SAEs die Textbeispiele findet, die es am stärksten aktivieren. Dies ist die primäre Methode, um Features zu interpretieren. | **Mittel** | **Hoch** |
| **B4: SAE Model Hub / Management** | Implementiere ein System zum Speichern und Laden von trainierten SAEs. Jeder SAE muss klar mit dem Basismodell und dem Layer verknüpft sein, auf dem er trainiert wurde (z.B. `llama3-8b_layer16_sae.pt`). | **Mittel** | **Mittel** |
| **B5: Abstraction Tracking Engine** | Entwickle eine Funktion, die für einen Input-Satz die Aktivierungsvektoren eines Tokens über alle Layer extrahiert und deren Kosinus-Ähnlichkeit zu einer Liste von vordefinierten Konzept-Vektoren (z.B. aus CAVs) berechnet. | **Mittel** | **Mittel** |

---

#### **2. CLI-Integration**

Wir schaffen die Kommandozeilen-Werkzeuge, um SAEs zu trainieren und die neuen Analysen durchzuführen.

| Aufgabe | Beschreibung | Komplexität | Priorität |
| :--- | :--- | :--- | :--- |
| **C1: CLI-Befehl `sae:train`** | Erstelle einen Befehl zum Starten der SAE-Trainingspipeline (`B1`). Parameter: Modell, Layer, Trainings-Dataset, Hyperparameter, Ausgabe-Pfad. | **Mittel** | **Hoch** |
| **C2: CLI-Befehl `sae:analyze-features`** | Erstelle einen Befehl, der die Analyse aus `B3` durchführt und eine interpretierbare Liste der Features und ihrer Top-Beispiele ausgibt. Z.B. `... --sae-model ... --top-k 10`. | **Mittel** | **Hoch** |
| **C3: CLI-Befehl `analyze:abstraction`** | Erstelle einen Befehl, der die Abstraktions-Analyse aus `B5` für einen gegebenen Prompt und eine Liste von Konzepten durchführt und die Ähnlichkeitswerte pro Layer als Tabelle ausgibt. | **Mittel** | **Mittel** |
| **C4: JSON-Output für alle Befehle** | Stelle sicher, dass alle neuen Befehle eine `--output json` Option haben, um strukturierte, maschinenlesbare Ergebnisse für den Agenten-Workflow zu liefern. | **Einfach** | **Hoch** |

---

#### **3. Web-Interface (UI/UX)**

Die Visualisierung macht die abstrakten Features und deren Verhalten greifbar.

| Aufgabe | Beschreibung | Komplexität | Priorität |
| :--- | :--- | :--- | :--- |
| **W1: API-Endpunkte für SAEs & Abstraktion** | Erstelle die notwendigen API-Endpunkte, um SAE-Feature-Informationen abzurufen und Abstraktions-Analysen aus dem Frontend anzustoßen. | **Mittel** | **Mittel** |
| **W2: "SAE Feature Explorer" UI** | Entwickle eine neue Dashboard-Ansicht, die als durchsuchbares "Wörterbuch" der gelernten Features dient. Zeigt für jedes Feature seine Top-aktivierenden Beispiele an. | **Mittel** | **Mittel** |
| **W3: Visualisierung von Feature-Aktivierungen** | Erweitere bestehende Heatmap-Visualisierungen, um optional statt der Neuron-Aktivierungen die (sehr spärlichen) SAE-Feature-Aktivierungen für einen bestimmten Prompt anzuzeigen. | **Mittel** | **Niedrig** |
| **W4: "Abstraction Trajectory" Plot** | Erstelle einen dedizierten Linien-Chart zur Visualisierung der Abstraktions-Analyse (`B5`). X-Achse: Layer-Nummer, Y-Achse: Kosinus-Ähnlichkeit. Jede Linie repräsentiert ein Konzept. | **Einfach** | **Mittel** |

---

### **Abschluss des ersten Hauptziels**

Mit der Umsetzung dieses Blocks haben wir einen kompletten, extrem leistungsfähigen Werkzeugkasten geschaffen, um von der reinen Beobachtung über kausale Eingriffe und Circuit-Analysen bis hin zur fundamentalen Dekodierung der "Sprache" des Modells vorzudringen.

Wir haben nun eine solide Grundlage, um uns dem nächsten strategischen Hauptziel zuzuwenden, z.B. **"Wie du das Projekt selbst noch besser machen kannst (Community, Automation, UX)"**.

Bereit für den nächsten großen Schritt, wenn du es bist!













































Absolut. Wir beginnen mit dem Aufbau des Ökosystems. Dieser Block ist weniger wissenschaftlich-experimentell, dafür aber entscheidend für den langfristigen Erfolg und die Akzeptanz des Projekts. Er ist stark von klassischer Software- und Infrastrukturentwicklung geprägt.

Hier ist die detaillierte Aufgabenliste für den Feature-Block **"'Community & Kollaboration: Der Analysis Zoo'"**.

---

### **Aufgabenliste: "Der Analysis Zoo"**

#### **1. Backend & Speicher-Infrastruktur**

Das Fundament der Plattform. Wir definieren, wie und wo die geteilten Analyse-Artefakte gespeichert und verwaltet werden.

| Aufgabe | Beschreibung | Komplexität | Priorität |
| :--- | :--- | :--- | :--- |
| **B1: Definition des "Artefakt"-Metadaten-Schemas** | Definiere ein `artefact.json` Manifest. Dieses beschreibt ein geteiltes Artefakt: Typ (SAE, Circuit, Config), Name, Version, Beschreibung, Autor, Basismodell, Dateipfade etc. | **Mittel** | **Hoch** |
| **B2: API-Server für den "Zoo"** | Entwerfe und implementiere einen zentralen API-Server (z.B. mit FastAPI), der die Metadaten aller Artefakte verwaltet. Er bietet Endpunkte zum Registrieren, Suchen und Abrufen von Artefakten. | **Schwer** | **Hoch** |
| **B3: Authentifizierungssystem (API-Keys)** | Implementiere ein einfaches, Token-basiertes Authentifizierungssystem für den API-Server, um festzulegen, wer Artefakte hochladen (`push`) darf. | **Mittel** | **Hoch** |
| **B4: Speicher-Backend (S3-kompatibel)** | Implementiere die Logik zum Hoch- und Herunterladen der eigentlichen Artefakt-Dateien (z.B. Model-Weights) in einen S3-kompatiblen Objektspeicher. Der API-Server sollte Pre-signed URLs für sichere Uploads generieren. | **Mittel** | **Hoch** |

---

#### **2. CLI-Integration**

Die primäre Schnittstelle für Power-User und Agenten, um mit dem "Zoo" zu interagieren.

| Aufgabe | Beschreibung | Komplexität | Priorität |
| :--- | :--- | :--- | :--- |
| **C1: `neuronmap login` Befehl** | Erstelle einen CLI-Befehl, um den API-Key des Nutzers sicher lokal zu speichern und für nachfolgende Befehle zu verwenden. | **Einfach** | **Hoch** |
| **C2: `neuronmap push` Befehl** | Implementiere den Befehl zum Hochladen eines Artefakts. Er packt die lokalen Dateien zusammen mit dem `artefact.json`, kommuniziert mit der API (`B2`) und lädt die Daten in den Speicher (`B4`). | **Schwer** | **Hoch** |
| **C3: `neuronmap pull` Befehl** | Implementiere den Befehl zum Herunterladen eines Artefakts. Er ruft die Metadaten von der API ab und lädt die zugehörigen Dateien in einen lokalen Cache-Ordner. Z.B. `neuronmap pull community/gpt2-induction-heads-v1`. | **Mittel** | **Hoch** |
| **C4: `neuronmap search` Befehl** | Implementiere einen Befehl zur Suche nach Artefakten im Zoo, mit Filtern nach Typ, Modell, etc. Gibt eine Liste passender Artefakt-IDs zurück. | **Mittel** | **Mittel** |

---

#### **3. Web-Interface (UI/UX)**

Das Schaufenster des "Zoos", das zum Stöbern, Entdecken und Nutzen der Community-Beiträge einlädt.

| Aufgabe | Beschreibung | Komplexität | Priorität |
| :--- | :--- | :--- | :--- |
| **W1: "Zoo" API-Integration im Frontend** | Verbinde das Web-Dashboard mit den neuen API-Endpunkten des "Zoo"-Servers, um die Artefakt-Daten abzurufen. | **Mittel** | **Mittel** |
| **W2: Artefakt-Galerie & Suchseite** | Erstelle eine neue Hauptseite "Zoo" oder "Hub", die alle geteilten Artefakte in einer durchsuchbaren und filterbaren Galerie (z.B. als Karten) anzeigt. | **Mittel** | **Hoch** |
| **W3: Artefakt-Detailseite** | Erstelle eine dedizierte Seite für jedes Artefakt. Sie zeigt alle Metadaten, eine gerenderte `README.md`, die Dateiliste und Code-Snippets zur einfachen Nutzung (z.B. den `pull`-Befehl). | **Mittel** | **Mittel** |
| **W4: Nutzerprofile (optional)** | Erstelle einfache Profilseiten für Nutzer, die alle von ihnen beigetragenen Artefakte auflisten. | **Mittel** | **Niedrig** |

---

### **Zusammenfassung und nächste Schritte**

Dieser Block ist ein fundamentaler Schritt, um NeuronMap von einem isolierten Werkzeug zu einer lebendigen, kollaborativen Plattform zu machen.

* **Startpunkt:** Die Backend-Infrastruktur (`B1-B4`) ist die absolute Voraussetzung. Parallel können die Kern-CLI-Befehle (`C1-C3`) entwickelt werden. Ein erster funktionierender `push` und `pull` Zyklus ist der wichtigste Meilenstein.
* **Agenten-Fokus:** Die klar definierte API und die CLI-Befehle sind perfekt für einen agenten-gestützten Workflow. Ein Agent könnte automatisch die neuesten und am besten bewerteten SAEs für seine Analysen `pullen`.

Wenn dieser Block steht, hat das Projekt ein starkes Fundament für Netzwerkeffekte. Als Nächstes könnten wir uns der **"Automation & Insight Mining"** zuwenden, um den Nutzern (und Agenten) proaktiv bei der Analyse zu helfen. Bereit, wenn du es bist.


























Hier ist der optimierte und korrigierte Plan für den Feature-Block **"Community & Kollaboration: Der Analysis Zoo"** mit Fokus auf technische Konsistenz und Umsetzbarkeit:

---

### **Korrektur & Optimierung des Plans**

#### **1. Backend & Speicher-Infrastruktur**
| Aufgabe | Korrektur/Optimierung | Priorität |
| :--- | :--- | :--- |
| **B1: Artefakt-Metadaten-Schema** | Schema-Erweiterung:<br>- `uuid` für eindeutige Identifikation<br>- `license` (MIT/Apache/CC-BY)<br>- `dependencies` (Python-Pakete)<br>- `created_at` Zeitstempel | **Hoch** |
| **B2: API-Server** | **Architektur-Änderung:**<br>- Verwendung von **GraphQL** (Apollo Server) statt REST für flexible Abfragen<br>- Integration von **ElasticSearch** für Volltextsuche | **Hoch** |
| **B3: Authentifizierung** | **Ersetzung durch:**<br>- OAuth2.0 (GitHub/GitLab) + JWT<br>- Rollen: `read`, `push`, `admin` | **Hoch** |
| **B4: Speicher-Backend** | **Optimierung:**<br>- Dedizierte Bucket-Struktur: `/{user_id}/{artifact_type}/{uuid}/`<br>- Automatische Virenscans via AWS Lambda<br>- Versionierung durch S3 Object Lock | **Hoch** |

#### **2. CLI-Integration**
| Aufgabe | Korrektur/Optimierung | Priorität |
| :--- | :--- | :--- |
| **C1: `login`** | **Ersetzung durch:**<br>`neuronmap auth login` (OAuth-Flow im Browser) | **Mittel** |
| **C2: `push`** | **Ergänzung:**<br>- Automatische Generierung von `artefact.json` aus Git-Metadaten<br>- Prä-Checks: Dateigröße (<5GB), Lizenz-Pflichtfeld | **Hoch** |
| **C3: `pull`** | **Optimierung:**<br>- Delta-Downloads (nur geänderte Dateien)<br>- Lokaler Cache mit TTL (7 Tage) | **Hoch** |
| **C4: `search`** | **Erweiterung:**<br>`neuronmap search --type=SAE --model=Llama-2 --license=MIT` | **Mittel** |

#### **3. Web-Interface (UI/UX)**
| Aufgabe | Korrektur/Optimierung | Priorität |
| :--- | :--- | :--- |
| **W1: API-Integration** | **Technologie:**<br>- GraphQL Client (Apollo Client) mit Caching | **Mittel** |
| **W2: Artefakt-Galerie** | **Funktionale Erweiterung:**<br>- Stern-Bewertungen (1-5)<br>- Download-Count Tracking<br>- "Verified"-Badge für offizielle Beiträge | **Hoch** |
| **W3: Detailseite** | **Neue Features:**<br>- Interaktive Preview für Circuits<br>- "Kopieren"-Button für `pull`-Befehle<br>- Abhängigkeitsbaum-Diagramm | **Mittel** |
| **W4: Nutzerprofile** | **Aufwertung:**<br>- Integration in W2/W3<br>- Anzeige von Reputation-Punkten<br>- **Neue Priorität: Hoch** | **Hoch** |

---

### **Kritische Architektur-Änderungen**
1. **Identifikation von Artefakten:**
   - Statt `community/gpt2-induction-heads-v1` → **UUID-basiert** (`#a1b2c3`)
   - Menschenlesbare Aliase über `name`-Feld im Schema

2. **Datei-Handling:**
   ```mermaid
   sequenceDiagram
     CLI->>API: POST /artefacts (Metadaten)
     API->>S3: Generate Pre-Signed URL
     API-->>CLI: Return Upload-URL
     CLI->>S3: Direktes Upload (Multipart)
     S3-->>API: Webhook bei Fertigstellung
     API->>DB: Mark as "READY"
   ```

3. **Sicherheit:**
   - Content-Security-Policy (CSP) für Web-UI
   - API-Rate-Limiting (100 req/min pro Nutzer)
   - Automatische Secrets-Rotation alle 90 Tage

---

### **Überarbeitete Prioritäten & Zeitplan**
| Phase | Aufgaben | Dauer |
| :--- | :--- | :--- |
| **Sprint 1** | B1, B3, C1, C2 (MVP für Push/Pull) | 2 Wochen |
| **Sprint 2** | B2, B4, C3, W1, W2 | 3 Wochen |
| **Sprint 3** | C4, W3, W4, ElasticSearch-Integration | 2 Wochen |
| **Polishing** | Sicherheitsaudit, Load-Testing, Dokumentation | 1 Woche |

---

### **Nächste Schritte**
1. **Starten mit Sprint 1:**
   - Implementierung des Metadaten-Schemas (`artefact.json` V1.0)
   - OAuth2.0-Integration (GitHub als First-Party)
   - Basis-CLI für `auth` und `push`

2. **Parallel:**
   - Einrichtung des S3-Buckets mit Versionierung
   - CI/CD-Pipeline für API-Server (Terraform + GitLab CI)

3. **Agenten-Integration:**
   ```python
   # Beispiel: Agent nutzt Analysis Zoo
   def load_artifact(artifact_id):
       if not local_cache.exists(artifact_id):
           run(f"neuronmap pull {artifact_id}")
       return load_from_cache(artifact_id)
   ```

Der Fokus liegt auf **standardkonformen Protokollen** (OAuth2, GraphQL, S3) und **skalierbarer Architektur**. Die Änderungen ermöglichen eine nahtlose Integration in ML-Workflows und erhöhen die Sicherheit.