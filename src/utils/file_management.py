"""
File Management Utilities for NeuronMap
======================================

This module provides comprehensive file management, data handling,
and compression utilities for the NeuronMap framework.
"""

import os
import shutil
import gzip
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tempfile

logger = logging.getLogger(__name__)


class FileManager:
    """File management utilities."""

    def __init__(self, base_dir: str = "."):
        """Initialize file manager.

        Args:
            base_dir: Base directory for file operations
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def create_file(self, filename: str, content: str = "") -> bool:
        """Create a file with content.

        Args:
            filename: Name of file to create
            content: Content to write to file

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.base_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to create file {filename}: {e}")
            return False

    def read_file(self, filename: str) -> Optional[str]:
        """Read content from a file.

        Args:
            filename: Name of file to read

        Returns:
            File content or None if failed
        """
        try:
            file_path = self.base_dir / filename
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {filename}: {e}")
            return None

    def delete_file(self, filename: str) -> bool:
        """Delete a file.

        Args:
            filename: Name of file to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.base_dir / filename
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {filename}: {e}")
            return False

    def list_files(self, pattern: str = "*") -> List[str]:
        """List files matching pattern.

        Args:
            pattern: Glob pattern to match files

        Returns:
            List of matching filenames
        """
        try:
            return [str(p.relative_to(self.base_dir)) for p in self.base_dir.glob(pattern)]
        except Exception as e:
            logger.error(f"Failed to list files with pattern {pattern}: {e}")
            return []

    def compress_file(self, filename: str) -> bool:
        """Compress a file using gzip.

        Args:
            filename: Name of file to compress

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.base_dir / filename
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')

            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            return True
        except Exception as e:
            logger.error(f"Failed to compress file {filename}: {e}")
            return False

    def decompress_file(self, filename: str) -> bool:
        """Decompress a gzipped file.

        Args:
            filename: Name of compressed file

        Returns:
            True if successful, False otherwise
        """
        try:
            compressed_path = self.base_dir / filename
            if not filename.endswith('.gz'):
                return False

            decompressed_path = compressed_path.with_suffix('')

            with gzip.open(compressed_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            return True
        except Exception as e:
            logger.error(f"Failed to decompress file {filename}: {e}")
            return False


class DataFileHandler:
    """Specialized handler for data files."""

    def __init__(self, data_dir: str = "data"):
        """Initialize data file handler.

        Args:
            data_dir: Directory for data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def save_json(self, data: Any, filename: str) -> bool:
        """Save data as JSON file.

        Args:
            data: Data to save
            filename: Name of JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.data_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON to {filename}: {e}")
            return False

    def load_json(self, filename: str) -> Optional[Any]:
        """Load data from JSON file.

        Args:
            filename: Name of JSON file

        Returns:
            Loaded data or None if failed
        """
        try:
            file_path = self.data_dir / filename
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {filename}: {e}")
            return None

    def save_pickle(self, data: Any, filename: str) -> bool:
        """Save data as pickle file.

        Args:
            data: Data to save
            filename: Name of pickle file

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.data_dir / filename
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save pickle to {filename}: {e}")
            return False

    def load_pickle(self, filename: str) -> Optional[Any]:
        """Load data from pickle file.

        Args:
            filename: Name of pickle file

        Returns:
            Loaded data or None if failed
        """
        try:
            file_path = self.data_dir / filename
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load pickle from {filename}: {e}")
            return None

    def save_csv_data(self, data: List[Dict], filename: str) -> bool:
        """Save data as CSV file.

        Args:
            data: List of dictionaries to save
            filename: Name of CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            file_path = self.data_dir / filename
            df.to_csv(file_path, index=False)
            return True
        except ImportError:
            logger.error("pandas not available for CSV operations")
            return False
        except Exception as e:
            logger.error(f"Failed to save CSV to {filename}: {e}")
            return False

    def load_csv_data(self, filename: str) -> Optional[List[Dict]]:
        """Load data from CSV file.

        Args:
            filename: Name of CSV file

        Returns:
            List of dictionaries or None if failed
        """
        try:
            import pandas as pd
            file_path = self.data_dir / filename
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        except ImportError:
            logger.error("pandas not available for CSV operations")
            return None
        except Exception as e:
            logger.error(f"Failed to load CSV from {filename}: {e}")
            return None


class ConfigValidator:
    """Configuration validation utilities."""

    def __init__(self):
        """Initialize config validator."""
        self.validation_errors = []

    def validate_config_file(self, config_path: str) -> bool:
        """Validate a configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            True if valid, False otherwise
        """
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    yaml.safe_load(f)
                else:
                    # Assume it's a text config
                    f.read()
            return True
        except Exception as e:
            self.validation_errors.append(str(e))
            return False

    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.validation_errors.copy()


class DataValidator:
    """Data validation utilities."""

    def __init__(self):
        """Initialize data validator."""
        self.validation_results = {}

    def validate_data_format(self, data: Any, expected_type: type) -> bool:
        """Validate data format.

        Args:
            data: Data to validate
            expected_type: Expected data type

        Returns:
            True if valid, False otherwise
        """
        try:
            return isinstance(data, expected_type)
        except Exception:
            return False

    def validate_required_fields(self, data: Dict, required_fields: List[str]) -> bool:
        """Validate required fields in data.

        Args:
            data: Data dictionary
            required_fields: List of required field names

        Returns:
            True if all fields present, False otherwise
        """
        try:
            for field in required_fields:
                if field not in data:
                    return False
            return True
        except Exception:
            return False
