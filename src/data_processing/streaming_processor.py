"""Streaming and batch data processing utilities."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import multiprocessing as mp
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional heavy dependencies
    import aiofiles
    import h5py
except ImportError:  # pragma: no cover - graceful degradation
    aiofiles = None
    h5py = None

from .quality_manager import DataQualityManager, QuestionMetadata
from ..utils.batch_processor import BatchProcessor


logger = logging.getLogger(__name__)


DEFAULT_STREAM_CONFIG: Dict[str, Any] = {
    "batch_size": 64,
    "buffer_size": 1024,
    "num_workers": max(mp.cpu_count() // 2, 1),
    "memory_limit_mb": None,
}


class StreamingDataProcessor:
    """Process large datasets either asynchronously or via lightweight helpers."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        max_workers: Optional[int] = None,
    ) -> None:
        config_dict = dict(DEFAULT_STREAM_CONFIG)
        if isinstance(config, dict):
            config_dict.update(config)

        if chunk_size != 1000 and "batch_size" not in config_dict:
            config_dict["batch_size"] = chunk_size
        if max_workers is not None:
            config_dict["num_workers"] = max_workers

        self.config = config_dict
        self.batch_size = int(config_dict.get("batch_size", 64))
        self.buffer_size = int(config_dict.get("buffer_size", max(self.batch_size * 2, 64)))
        self.max_workers = int(config_dict.get("num_workers") or mp.cpu_count())
        self.memory_limit_mb = config_dict.get("memory_limit_mb")
        self.chunk_size = max(self.batch_size, chunk_size)
        self.quality_manager = DataQualityManager()

    # ------------------------------------------------------------------
    # Asynchronous question processing helpers used in the original API
    # ------------------------------------------------------------------
    async def process_questions_stream(
        self,
        input_file: str,
        output_file: str,
        validation: bool = True,
    ) -> Dict[str, Any]:
        if h5py is None or aiofiles is None:
            raise ImportError("h5py and aiofiles required for streaming processing")

        stats = {
            "total_processed": 0,
            "valid_questions": 0,
            "invalid_questions": 0,
            "duplicates_found": 0,
            "processing_time": 0,
        }

        start_time = time.time()

        async with aiofiles.open(input_file, "r") as handle:
            with h5py.File(output_file, "w") as hfile:
                questions_group = hfile.create_group("questions")
                metadata_group = hfile.create_group("metadata")

                chunk: List[Tuple[int, Dict[str, Any]]] = []
                question_index = 0

                async for line in handle:
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError as exc:
                        logger.warning("Invalid JSON at line %d: %s", question_index, exc)
                        stats["invalid_questions"] += 1
                        question_index += 1
                        continue

                    chunk.append((question_index, data))
                    question_index += 1

                    if len(chunk) >= self.chunk_size:
                        await self._process_chunk(
                            chunk,
                            questions_group,
                            metadata_group,
                            stats,
                            validation,
                        )
                        chunk = []

                if chunk:
                    await self._process_chunk(
                        chunk,
                        questions_group,
                        metadata_group,
                        stats,
                        validation,
                    )

        stats["processing_time"] = time.time() - start_time
        return stats

    async def _process_chunk(
        self,
        chunk: Sequence[Tuple[int, Dict[str, Any]]],
        questions_group,
        metadata_group,
        stats: Dict[str, Any],
        validation: bool,
    ) -> None:
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    self._process_single_question,
                    idx,
                    data,
                    validation,
                )
                for idx, data in chunk
            ]
            results = await asyncio.gather(*tasks)

        for idx, data, validation_result in results:
            if validation_result and validation_result["valid"]:
                questions_group[f"question_{idx}"] = data["text"].encode("utf-8")
                metadata = QuestionMetadata(
                    id=data.get("id", str(uuid.uuid4())),
                    text=data["text"],
                    category=data.get("category", "unknown"),
                    difficulty=data.get("difficulty", 5),
                    language=data.get("language", "en"),
                    generation_time=data.get("generation_time", 0.0),
                    model_used=data.get("model_used", "unknown"),
                    prompt_template=data.get("prompt_template", ""),
                    hash=hashlib.sha256(data["text"].encode()).hexdigest(),
                    created_at=data.get("created_at", str(time.time())),
                )
                metadata_group[f"metadata_{idx}"] = json.dumps(metadata.to_dict()).encode("utf-8")
                stats["valid_questions"] += 1
            else:
                stats["invalid_questions"] += 1
            stats["total_processed"] += 1

    def _process_single_question(
        self,
        idx: int,
        data: Dict[str, Any],
        validation: bool,
    ) -> Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]:
        validation_result = None
        if validation:
            validation_result = self.quality_manager.validate_question(data.get("text", ""))
        return idx, data, validation_result

    def process_questions_batch(
        self,
        input_file: str,
        output_file: str,
        validation: bool = True,
    ) -> Dict[str, Any]:
        stats = {
            "total_processed": 0,
            "valid_questions": 0,
            "invalid_questions": 0,
            "processing_time": 0,
        }

        start_time = time.time()
        valid_questions: List[Dict[str, Any]] = []

        with open(input_file, "r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle):
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as exc:
                    logger.warning("Invalid JSON at line %d: %s", line_num, exc)
                    stats["invalid_questions"] += 1
                    continue

                if validation:
                    if self.quality_manager.validate_question(data.get("text", "")).get("valid"):
                        valid_questions.append(data)
                        stats["valid_questions"] += 1
                    else:
                        stats["invalid_questions"] += 1
                else:
                    valid_questions.append(data)
                    stats["valid_questions"] += 1

                stats["total_processed"] += 1

        with open(output_file, "w", encoding="utf-8") as outfile:
            for question in valid_questions:
                outfile.write(json.dumps(question) + "\n")

        stats["processing_time"] = time.time() - start_time
        return stats

    # ------------------------------------------------------------------
    # Lightweight helpers required by tests
    # ------------------------------------------------------------------
    def stream_data(self, file_path: str) -> Iterator[List[Dict[str, Any]]]:
        batch_size = max(1, self.batch_size)
        yield from self._iter_dataset(file_path, batch_size=batch_size)

    def stream_data_memory_efficient(self, file_path: str) -> Iterator[List[Dict[str, Any]]]:
        if self.memory_limit_mb is None:
            yield from self.stream_data(file_path)
            return

        approx_item_kb = 4  # heuristic suitable for tests
        max_items = max(1, int((self.memory_limit_mb * 1024) / approx_item_kb))
        batch_size = max(1, min(self.batch_size, max_items))
        yield from self._iter_dataset(file_path, batch_size=batch_size)

    def _iter_dataset(self, file_path: str, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(file_path)

        if path.suffix.lower() == ".jsonl":
            batch: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid JSON line in %s", path)
                        continue
                    batch.append(self._normalise_stream_item(data))
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            if batch:
                yield batch
            return

        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        items: Sequence[Any]
        if isinstance(data, list):
            items = data
        else:
            items = [data]

        normalised = [self._normalise_stream_item(item) for item in items]
        for index in range(0, len(normalised), batch_size):
            yield normalised[index : index + batch_size]

    def _normalise_stream_item(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        return {"text": str(item)}

