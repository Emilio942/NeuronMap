"""File handling utilities for NeuronMap."""

import json
import csv
import pickle
import gzip
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable
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


class FileManager:
    """High-level helper for working with files relative to a base directory."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str | Path) -> Path:
        path = self.base_dir / Path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative_path: str, data: Any, *, indent: int = 2) -> Path:
        target = self._resolve(relative_path)
        save_json(data, str(target), indent=indent)
        return target

    def read_json(self, relative_path: str) -> Any:
        target = self._resolve(relative_path)
        data = load_json(str(target))
        if data is None:
            raise FileNotFoundError(f"JSON file not found or invalid: {target}")
        return data

    def file_exists(self, relative_path: str) -> bool:
        return (self.base_dir / Path(relative_path)).exists()

    def compress_file(self, relative_path: str) -> str:
        source = self._resolve(relative_path)
        if not source.exists():
            raise FileNotFoundError(source)

        if source.suffix:
            target = source.with_suffix(source.suffix + ".gz")
        else:
            target = source.with_name(source.name + ".gz")

        with open(source, 'rb') as f_in, gzip.open(target, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        return str(target.relative_to(self.base_dir))

    def decompress_file(self, relative_path: str) -> str:
        compressed = self._resolve(relative_path)
        if not compressed.exists():
            raise FileNotFoundError(compressed)

        target_name = compressed.name[:-3] if compressed.suffix == '.gz' else compressed.stem
        target = compressed.with_name(target_name)

        with gzip.open(compressed, 'rb') as f_in, open(target, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        return str(target.relative_to(self.base_dir))


class DataFileHandler(FileManager):
    """Specialised helpers for structured dataset storage."""

    def __init__(self, data_dir: str | Path | None = None, **kwargs):
        if data_dir is not None and 'base_dir' not in kwargs:
            kwargs['base_dir'] = data_dir
        super().__init__(**kwargs)

    def write_csv(self, relative_path: str, rows: Iterable[Dict[str, Any]]) -> Path:
        rows = list(rows)
        target = self._resolve(relative_path)

        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []

        with open(target, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

        return target

    def read_csv(self, relative_path: str) -> List[Dict[str, Any]]:
        target = self._resolve(relative_path)
        if not target.exists():
            raise FileNotFoundError(target)

        with open(target, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    def write_hdf5(self, relative_path: str, datasets: Dict[str, Any]) -> Path:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required for HDF5 operations") from exc

        target = self._resolve(relative_path)
        with h5py.File(target, 'w') as h5file:
            for name, data in datasets.items():
                h5file.create_dataset(name, data=data)
        return target

    def read_hdf5(self, relative_path: str) -> Dict[str, Any]:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required for HDF5 operations") from exc

        target = self._resolve(relative_path)
        if not target.exists():
            raise FileNotFoundError(target)

        data: Dict[str, Any] = {}
        with h5py.File(target, 'r') as h5file:
            for name in h5file.keys():
                data[name] = h5file[name][()]
        return data
