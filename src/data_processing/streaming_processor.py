"""Streaming data processor for large datasets."""

import json
import logging
import time
import asyncio
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import uuid
import hashlib

try:
    import h5py
    import aiofiles
except ImportError:
    h5py = None
    aiofiles = None

from .quality_manager import DataQualityManager, QuestionMetadata


logger = logging.getLogger(__name__)


class StreamingDataProcessor:
    """Process large datasets in streaming fashion."""

    def __init__(self, chunk_size: int = 1000, max_workers: int = None):
        """Initialize streaming processor.

        Args:
            chunk_size: Size of data chunks to process.
            max_workers: Maximum number of worker threads.
        """
        self.chunk_size = chunk_size
        self.max_workers = max_workers or mp.cpu_count()
        self.quality_manager = DataQualityManager()

    async def process_questions_stream(self, input_file: str, output_file: str,
                                     validation: bool = True) -> Dict[str, Any]:
        """Process questions in streaming fashion.

        Args:
            input_file: Input JSONL file with questions.
            output_file: Output HDF5 file for processed data.
            validation: Whether to perform quality validation.

        Returns:
            Processing statistics.
        """
        if h5py is None or aiofiles is None:
            raise ImportError("h5py and aiofiles required for streaming processing")

        stats = {
            'total_processed': 0,
            'valid_questions': 0,
            'invalid_questions': 0,
            'duplicates_found': 0,
            'processing_time': 0
        }

        start_time = time.time()

        async with aiofiles.open(input_file, 'r') as f:
            with h5py.File(output_file, 'w') as hf:
                questions_group = hf.create_group('questions')
                metadata_group = hf.create_group('metadata')

                chunk_buffer = []
                question_index = 0

                async for line in f:
                    try:
                        data = json.loads(line.strip())
                        chunk_buffer.append((question_index, data))

                        if len(chunk_buffer) >= self.chunk_size:
                            await self._process_chunk(
                                chunk_buffer, questions_group, metadata_group,
                                stats, validation
                            )
                            chunk_buffer = []

                        question_index += 1

                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {question_index}: {e}")
                        stats['invalid_questions'] += 1

                # Process remaining chunk
                if chunk_buffer:
                    await self._process_chunk(
                        chunk_buffer, questions_group, metadata_group,
                        stats, validation
                    )

        stats['processing_time'] = time.time() - start_time
        return stats

    async def _process_chunk(self, chunk: List[Tuple[int, Dict[str, Any]]],
                           questions_group, metadata_group,
                           stats: Dict[str, Any], validation: bool):
        """Process a chunk of questions.

        Args:
            chunk: List of (index, question_data) tuples.
            questions_group: HDF5 group for questions.
            metadata_group: HDF5 group for metadata.
            stats: Statistics dictionary to update.
            validation: Whether to perform validation.
        """
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = []

            for idx, data in chunk:
                task = loop.run_in_executor(
                    executor, self._process_single_question,
                    idx, data, validation
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            # Store results
            for idx, data, validation_result in results:
                if validation_result and validation_result['valid']:
                    questions_group[f'question_{idx}'] = data['text'].encode('utf-8')

                    # Store metadata
                    metadata = QuestionMetadata(
                        id=data.get('id', str(uuid.uuid4())),
                        text=data['text'],
                        category=data.get('category', 'unknown'),
                        difficulty=data.get('difficulty', 5),
                        language=data.get('language', 'en'),
                        generation_time=data.get('generation_time', 0.0),
                        model_used=data.get('model_used', 'unknown'),
                        prompt_template=data.get('prompt_template', ''),
                        hash=hashlib.md5(data['text'].encode()).hexdigest(),
                        created_at=data.get('created_at', str(time.time()))
                    )

                    metadata_group[f'metadata_{idx}'] = json.dumps(metadata.to_dict()).encode('utf-8')
                    stats['valid_questions'] += 1
                else:
                    stats['invalid_questions'] += 1

                stats['total_processed'] += 1

    def _process_single_question(self, idx: int, data: Dict[str, Any],
                                validation: bool) -> Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Process a single question.

        Args:
            idx: Question index.
            data: Question data.
            validation: Whether to perform validation.

        Returns:
            Tuple of (index, data, validation_result).
        """
        validation_result = None

        if validation:
            question_text = data.get('text', '')
            validation_result = self.quality_manager.validate_question(question_text)

        return idx, data, validation_result

    def process_questions_batch(self, input_file: str, output_file: str,
                               validation: bool = True) -> Dict[str, Any]:
        """Process questions in batch mode (synchronous).

        Args:
            input_file: Input JSONL file with questions.
            output_file: Output file for processed data.
            validation: Whether to perform quality validation.

        Returns:
            Processing statistics.
        """
        stats = {
            'total_processed': 0,
            'valid_questions': 0,
            'invalid_questions': 0,
            'processing_time': 0
        }

        start_time = time.time()

        with open(input_file, 'r') as infile:
            valid_questions = []

            for line_num, line in enumerate(infile):
                try:
                    data = json.loads(line.strip())
                    question_text = data.get('text', '')

                    if validation:
                        validation_result = self.quality_manager.validate_question(question_text)
                        if validation_result['valid']:
                            valid_questions.append(data)
                            stats['valid_questions'] += 1
                        else:
                            stats['invalid_questions'] += 1
                    else:
                        valid_questions.append(data)
                        stats['valid_questions'] += 1

                    stats['total_processed'] += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    stats['invalid_questions'] += 1

        # Save valid questions
        with open(output_file, 'w') as outfile:
            for question_data in valid_questions:
                outfile.write(json.dumps(question_data) + '\n')

        stats['processing_time'] = time.time() - start_time
        logger.info(f"Processed {stats['total_processed']} questions in {stats['processing_time']:.2f}s")

        return stats
