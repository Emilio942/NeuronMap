"""File handling utilities for NeuronMap."""

import json
import csv
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging


logger = logging.getLogger(__name__)


def save_json(data: Any, filepath: str, indent: int = 2) -> bool:
    """Save data to JSON file.

    Args:
        data: Data to save.
        filepath: Output file path.
        indent: JSON indentation.

    Returns:
        True if successful.
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        logger.debug(f"JSON saved to {path}")
        return True

    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        return False


def load_json(filepath: str) -> Optional[Any]:
    """Load data from JSON file.

    Args:
        filepath: Input file path.

    Returns:
        Loaded data or None if error.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.debug(f"JSON loaded from {filepath}")
        return data

    except FileNotFoundError:
        logger.error(f"JSON file not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        return None


def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> bool:
    """Save data to JSON Lines file.

    Args:
        data: List of dictionaries to save.
        filepath: Output file path.

    Returns:
        True if successful.
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')

        logger.debug(f"JSONL saved to {path}")
        return True

    except Exception as e:
        logger.error(f"Error saving JSONL to {filepath}: {e}")
        return False


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load data from JSON Lines file.

    Args:
        filepath: Input file path.

    Returns:
        List of loaded dictionaries.
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")

        logger.debug(f"JSONL loaded from {filepath}: {len(data)} items")
        return data

    except FileNotFoundError:
        logger.error(f"JSONL file not found: {filepath}")
        return []
    except Exception as e:
        logger.error(f"Error loading JSONL from {filepath}: {e}")
        return []


def save_pickle(data: Any, filepath: str) -> bool:
    """Save data to pickle file.

    Args:
        data: Data to save.
        filepath: Output file path.

    Returns:
        True if successful.
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logger.debug(f"Pickle saved to {path}")
        return True

    except Exception as e:
        logger.error(f"Error saving pickle to {filepath}: {e}")
        return False


def load_pickle(filepath: str) -> Optional[Any]:
    """Load data from pickle file.

    Args:
        filepath: Input file path.

    Returns:
        Loaded data or None if error.
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        logger.debug(f"Pickle loaded from {filepath}")
        return data

    except FileNotFoundError:
        logger.error(f"Pickle file not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading pickle from {filepath}: {e}")
        return None


def ensure_directory(filepath: str) -> Path:
    """Ensure directory exists for a given filepath.

    Args:
        filepath: File path to create directory for.

    Returns:
        Path object of the directory.
    """
    path = Path(filepath)
    if path.suffix:  # It's a file path
        directory = path.parent
    else:  # It's a directory path
        directory = path

    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_files(directory: str, pattern: str = "*", recursive: bool = False) -> List[Path]:
    """List files in directory matching pattern.

    Args:
        directory: Directory to search.
        pattern: File pattern to match.
        recursive: Whether to search recursively.

    Returns:
        List of matching file paths.
    """
    path = Path(directory)

    if not path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    if recursive:
        files = list(path.rglob(pattern))
    else:
        files = list(path.glob(pattern))

    # Filter to only files (not directories)
    files = [f for f in files if f.is_file()]

    return sorted(files)
