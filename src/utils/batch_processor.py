"""Centralised batch processing utilities for NeuronMap.

This module exposes the canonical :class:`BatchProcessor` implementation used
across NeuronMap. It provides lightweight batching helpers for activation
analysis and question pipelines, plus optional asynchronous processing. Any
additional batch-oriented functionality should build on this module instead of
creating new processors.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Utility class for processing batched data.

    Parameters
    ----------
    config:
        Optional configuration dictionary. Supported keys include:

        ``batch_size``
            Number of samples per batch (default: 32)
        ``num_workers``
            Number of worker threads in parallel mode (default: half CPU cores)
        ``parallel_processing``
            Whether to enable thread-based parallelism (default: True)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.config = {
            "batch_size": int(config.get("batch_size", 32)),
            "num_workers": int(config.get("num_workers", max(mp.cpu_count() // 2, 1))),
            "parallel_processing": bool(config.get("parallel_processing", True)),
        }

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _split_batches(self, items: Sequence[Any], batch_size: Optional[int] = None) -> List[Sequence[Any]]:
        size = max(1, int(batch_size or self.config["batch_size"]))
        return [items[i : i + size] for i in range(0, len(items), size)]

    # ------------------------------------------------------------------
    # Activation utilities
    # ------------------------------------------------------------------
    def process_activation_batches(
        self,
        activations: np.ndarray,
        processing_func: Callable[[np.ndarray], np.ndarray],
        parallel: Optional[bool] = None,
    ) -> np.ndarray:
        """Process model activations in batches."""

        data = np.asarray(activations)
        if data.ndim < 2:
            raise ValueError("Activations must be at least 2D")

        chunks = self._split_batches(data)
        if not chunks:
            return np.array([])

        use_parallel = self.config["parallel_processing"] if parallel is None else parallel

        if use_parallel and len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=self.config["num_workers"]) as executor:
                results = list(executor.map(processing_func, chunks))
        else:
            results = [processing_func(chunk) for chunk in chunks]

        return np.concatenate(results, axis=0)

    # ------------------------------------------------------------------
    # Question-processing helpers (legacy compatibility)
    # ------------------------------------------------------------------
    def process_questions_batch(
        self,
        questions: Sequence[Any],
        process_fn: Callable[[Sequence[Any]], Iterable[Any]],
        **kwargs: Any,
    ) -> List[Any]:
        """Process question datasets in batches."""

        batches = self._split_batches(list(questions))
        logger.info("Processing %d questions in %d batches", len(questions), len(batches))

        results: List[Any] = []
        for index, batch in enumerate(batches, start=1):
            try:
                batch_result = process_fn(batch, **kwargs)
                if isinstance(batch_result, Iterable) and not isinstance(batch_result, (str, bytes)):
                    results.extend(list(batch_result))
                else:
                    results.append(batch_result)
                logger.debug("Completed batch %d/%d", index, len(batches))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Error processing batch %d: %s", index, exc)
        return results

    async def process_questions_async(
        self,
        questions: Sequence[Any],
        process_fn: Callable[[Any], Any],
        **kwargs: Any,
    ) -> List[Any]:
        """Asynchronously process questions with bounded concurrency."""

        semaphore = asyncio.Semaphore(self.config["num_workers"])

        async def process_single(question: Any) -> Any:
            async with semaphore:
                return await process_fn(question, **kwargs)

        tasks = [process_single(question) for question in questions]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        return [value for value in outputs if not isinstance(value, Exception)]

    # ------------------------------------------------------------------
    # Generic batch helper
    # ------------------------------------------------------------------
    def apply(
        self,
        items: Sequence[Any],
        process_fn: Callable[[Sequence[Any]], Iterable[Any]],
        batch_size: Optional[int] = None,
    ) -> List[Any]:
        """Generic helper applying ``process_fn`` to batches of ``items``."""

        return self.process_questions_batch(items, process_fn, batch_size=batch_size)  # type: ignore[arg-type]


__all__ = ["BatchProcessor"]


def _demo() -> None:  # pragma: no cover - simple usage example
    sample = np.random.randn(128, 16)
    processor = BatchProcessor({"batch_size": 32, "parallel_processing": False})
    means = processor.process_activation_batches(sample, lambda chunk: np.mean(chunk, axis=1, keepdims=True))
    print(f"Processed {means.shape[0]} samples")


if __name__ == "__main__":  # pragma: no cover
    _demo()
