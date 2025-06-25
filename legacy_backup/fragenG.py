# Block 1: Import notwendiger Bibliotheken
# import requests # Nicht mehr direkt benötigt
import ollama    # NEU: Importiere die offizielle Ollama-Bibliothek
import json
import time
import argparse
import sys
import os
from tqdm import tqdm
import logging
# NEU: Importiere spezifische Fehlerklassen aus ollama, falls benötigt (optional, aber gut für spezifisches Handling)
from ollama import ResponseError, RequestError # RequestError für Verbindungs-/Timeout-Probleme

# --- Konstanten und Konfiguration ---
DEFAULT_OLLAMA_HOST = "http://localhost:11434" # Standard-Host für Ollama
DEFAULT_MODEL_NAME = "deepseek-r1:32b"
DEFAULT_NUM_QUESTIONS_TARGET = 100000
DEFAULT_BATCH_SIZE = 20
DEFAULT_OUTPUT_FILE = "generated_questions.jsonl"
RETRY_DELAY_SECONDS = 10
MAX_RETRIES_PER_BATCH = 5

# --- Logging Konfiguration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Block 2: Funktion zur Kommunikation mit Ollama über die Bibliothek (STARK ÜBERARBEITET)
def call_ollama_generate(ollama_host: str, model_name: str, prompt: str) -> str | None:
    """
    Sendet einen Prompt an Ollama über die offizielle Python-Bibliothek.

    Args:
        ollama_host (str): Die Host-Adresse deines Ollama-Servers (z.B. "http://localhost:11434").
        model_name (str): Der Name des zu verwendenden Ollama-Modells.
        prompt (str): Der Prompt, der an das Modell gesendet wird.

    Returns:
        str | None: Die generierte Textantwort des Modells oder None bei einem Fehler.
    """
    try:
        # Erstelle einen Client, der auf den spezifischen Host zeigt
        client = ollama.Client(host=ollama_host)

        # Rufe die generate-Funktion auf
        response = client.generate(
            model=model_name,
            prompt=prompt,
            # stream=False ist hier Standard, wenn nicht anders angegeben
            # options={'temperature': 0.7} # Beispiel für weitere Optionen
        )

        # Gib den generierten Text zurück
        return response.get("response", "").strip()

    # Fehlerbehandlung spezifisch für die ollama Bibliothek
    except ResponseError as e:
        # Fehler, die von der Ollama API zurückgegeben werden (z.B. Modell nicht gefunden)
        logger.error(f"Ollama API Fehler: {e.error} (Statuscode: {e.status_code})")
        if "model" in e.error.lower() and "not found" in e.error.lower():
             logger.error(f"Stelle sicher, dass das Modell '{model_name}' mit 'ollama pull {model_name}' heruntergeladen wurde.")
        return None
    except RequestError as e:
         # Fehler beim Verbindungsaufbau oder Timeout (abgeleitet von httpx)
         logger.error(f"Verbindungsfehler zu Ollama unter {ollama_host}: {e}")
         logger.error("Stelle sicher, dass Ollama läuft und unter der angegebenen Adresse erreichbar ist.")
         return None
    except Exception as e:
        # Fange andere unerwartete Fehler ab
        logger.error(f"Ein unerwarteter Fehler bei der Kommunikation mit Ollama ist aufgetreten: {e}", exc_info=True)
        return None


# Block 3: Funktion zum Extrahieren der Fragen aus der Antwort (unverändert)
def parse_questions_from_response(response_text: str) -> list[str]:
    """
    Extrahiert einzelne Fragen aus dem vom LLM generierten Textblock.
    Nimmt an, dass jede Frage in einer neuen Zeile steht (vom Prompt erzwungen).
    Minimale Bereinigung.
    """
    questions = []
    if not response_text:
        return questions

    lines = response_text.split('\n')
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:
            questions.append(cleaned_line)
    return questions

# Block 4: Funktion zum Speichern der Fragen in einer Datei (unverändert)
def save_questions_to_file(questions: list[str], filename: str) -> bool:
    """
    Hängt eine Liste von Fragen an eine Datei im JSON Lines Format an.
    Gibt True bei Erfolg zurück, False bei Fehler.
    """
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            for question in questions:
                json_record = json.dumps({"question": question})
                f.write(json_record + '\n')
        return True
    except IOError as e:
        logger.error(f"Konnte nicht in die Datei {filename} schreiben: {e}")
        return False
    except Exception as e:
        logger.error(f"Ein unerwarteter Fehler beim Schreiben in {filename} ist aufgetreten: {e}")
        return False

# Block 5: Funktion zum Zählen der bereits vorhandenen Fragen in der Datei (unverändert)
def count_existing_questions(filename: str) -> int:
    """
    Zählt die Anzahl der Zeilen (und damit Fragen) in einer JSONL-Datei.
    Gibt 0 zurück, wenn die Datei nicht existiert oder ein Fehler auftritt.
    """
    if not os.path.exists(filename):
        return 0
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        logger.info(f"{count} bereits vorhandene Fragen in '{filename}' gefunden.")
        return count
    except IOError as e:
        logger.error(f"Fehler beim Lesen der Datei {filename} zum Zählen der Zeilen: {e}")
        return 0
    except Exception as e:
        logger.error(f"Unerwarteter Fehler beim Zählen der Zeilen in {filename}: {e}")
        return 0

# Block 6: Hauptlogik des Skripts (nur Aufruf von call_ollama_generate angepasst)
def main(args):
    """
    Hauptfunktion zum Generieren der Fragen mit Fortsetzungs- und Retry-Logik.
    """
    logger.info(f"Starte die Generierung von Fragen...")
    logger.info(f"Ziel: {args.num_questions} Fragen")
    logger.info(f"Verwendetes Modell: {args.model}")
    # logger.info(f"Ollama URL: {args.ollama_url}") # Alt
    logger.info(f"Ollama Host: {args.ollama_host}") # NEU
    logger.info(f"Ausgabedatei: {args.output_file} (wird angehängt, falls vorhanden)")
    logger.info(f"Batch-Größe: {args.batch_size}")
    logger.info(f"Max. Retries pro Batch: {args.max_retries}") # Verwende args direkt
    logger.info("-" * 30)

    questions_generated_count = count_existing_questions(args.output_file)

    if questions_generated_count >= args.num_questions:
        logger.info(f"Ziel von {args.num_questions} Fragen bereits in '{args.output_file}' erreicht oder überschritten ({questions_generated_count} vorhanden). Beende.")
        return

    with tqdm(total=args.num_questions, initial=questions_generated_count, desc="Generiere Fragen", unit=" Frage") as pbar:
        consecutive_batch_failures = 0

        while questions_generated_count < args.num_questions:
            num_to_generate_this_batch = min(args.batch_size, args.num_questions - questions_generated_count)

            prompt = (
                f"Generate exactly {num_to_generate_this_batch} diverse and unique questions. "
                f"Topics should include: science, history, philosophy, technology, ethics, creative writing, daily life, hypothetical scenarios, abstract concepts. "
                f"Vary question complexity and style (e.g., open-ended, specific, comparative). "
                f"VERY IMPORTANT: Output ONLY the questions. Each question MUST be on a new line. "
                f"Do NOT include any introduction, conclusion, remarks, or numbering. Just the raw questions, one per line."
            )

            batch_success = False
            for attempt in range(args.max_retries): # Verwende args.max_retries
                logger.debug(f"Versuch {attempt + 1}/{args.max_retries} für Batch beginnend bei Frage {questions_generated_count + 1}")

                # Rufe die überarbeitete Funktion auf
                response_text = call_ollama_generate(args.ollama_host, args.model, prompt)

                if response_text:
                    new_questions = parse_questions_from_response(response_text)
                    if new_questions:
                        questions_to_save = new_questions[:num_to_generate_this_batch]
                        if save_questions_to_file(questions_to_save, args.output_file):
                            num_actually_added = len(questions_to_save)
                            questions_generated_count += num_actually_added
                            pbar.update(num_actually_added)
                            batch_success = True
                            consecutive_batch_failures = 0
                            logger.debug(f"{num_actually_added} Fragen erfolgreich hinzugefügt.")
                            break
                        else:
                            logger.error(f"Kritischer Fehler beim Speichern der Fragen in Batch {attempt + 1}. Breche Skript ab.")
                            sys.exit(1)
                    else:
                        logger.warning(f"Antwort von Ollama erhalten, aber keine Fragen konnten extrahiert werden (Versuch {attempt + 1}). Inhalt: '{response_text[:100]}...'")
                else:
                    logger.warning(f"Fehler bei der Ollama-Kommunikation (Versuch {attempt + 1}). Siehe vorherige Fehlermeldung für Details.")

                if attempt < args.max_retries - 1:
                    logger.info(f"Warte {RETRY_DELAY_SECONDS} Sekunden vor dem nächsten Versuch...")
                    time.sleep(RETRY_DELAY_SECONDS)

            if not batch_success:
                consecutive_batch_failures += 1
                logger.error(f"Batch konnte nach {args.max_retries} Versuchen nicht generiert oder verarbeitet werden.")
                if consecutive_batch_failures >= 3:
                     logger.critical(f"Mehrere aufeinanderfolgende Batches fehlgeschlagen ({consecutive_batch_failures}). Breche Skript ab, um Endlosschleife bei persistenten Problemen zu verhindern.")
                     sys.exit(1)
                else:
                     logger.warning("Überspringe diesen Batch und versuche den nächsten.")

    logger.info("\n" + "=" * 30)
    if questions_generated_count >= args.num_questions:
        logger.info(f"Generierung abgeschlossen!")
        logger.info(f"Insgesamt {questions_generated_count} Fragen befinden sich nun in '{args.output_file}'.")
    else:
        logger.warning(f"Generierung beendet, aber Ziel von {args.num_questions} nicht erreicht ({questions_generated_count} Fragen generiert).")
    logger.info("=" * 30)
    logger.info("Hinweis: Die generierten Fragen enthalten möglicherweise Duplikate oder sind nicht perfekt formatiert. Eine Nachbearbeitung wird empfohlen.")


# Block 7: Kommandozeilenargumente verarbeiten und Skript starten (Argument --ollama-host angepasst)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generiert eine große Anzahl von Fragen mit einem lokalen LLM über die offizielle Ollama Python-Bibliothek. Setzt die Generierung fort, falls die Ausgabedatei bereits existiert.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Name des Ollama-Modells")
    parser.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS_TARGET, help="Gesamtzahl der zu generierenden Fragen")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Anzahl Fragen pro API-Aufruf")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE, help="Pfad zur Ausgabedatei (.jsonl Format, wird angehängt)")
    # parser.add_argument("--ollama-url", ...) # ALT
    parser.add_argument( # NEU
        "--ollama-host",
        type=str,
        default=DEFAULT_OLLAMA_HOST,
        help="Host-Adresse des Ollama-Servers (z.B. 'http://localhost:11434' oder 'http://192.168.1.10:11434')"
    )
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES_PER_BATCH, help="Maximale Wiederholungsversuche pro fehlgeschlagenem Batch")

    args = parser.parse_args()

    # Die MAX_RETRIES_PER_BATCH Konstante wird nicht mehr global überschrieben,
    # stattdessen wird direkt args.max_retries in der main-Schleife verwendet.

    main(args)
