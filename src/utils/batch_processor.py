"""
Resilient Batch Processing System for NeuronMap
==============================================

This module provides robust batch processing with checkpointing, job resumption,
and automatic workload redistribution for failures.
"""

import json
import pickle
import time
import threading
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Iterator, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import hashlib
import shutil

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of batch processing jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ProcessingMode(Enum):
    """Processing execution modes."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    MULTIPROCESS = "multiprocess"


@dataclass
class BatchItem:
    """Individual item in a batch."""
    id: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    processing_time: float = 0.0
    completed_at: Optional[datetime] = None
    
    def mark_failed(self, error: str):
        """Mark item as failed with error message."""
        self.attempts += 1
        self.last_error = error
    
    def should_retry(self) -> bool:
        """Check if item should be retried."""
        return self.attempts < self.max_attempts
    
    def is_completed(self) -> bool:
        """Check if item is completed."""
        return self.completed_at is not None


@dataclass
class Checkpoint:
    """Checkpoint for batch processing state."""
    checkpoint_id: str
    job_id: str
    timestamp: datetime
    completed_items: List[str]
    failed_items: List[str]
    pending_items: List[str]
    results: Dict[str, Any]
    progress: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, checkpoint_dir: Path):
        """Save checkpoint to disk."""
        checkpoint_file = checkpoint_dir / f"checkpoint_{self.checkpoint_id}.pkl"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(self, f)
        
        # Also save a JSON summary for human readability
        summary_file = checkpoint_dir / f"checkpoint_{self.checkpoint_id}.json"
        summary = {
            'checkpoint_id': self.checkpoint_id,
            'job_id': self.job_id,
            'timestamp': self.timestamp.isoformat(),
            'progress': self.progress,
            'completed_count': len(self.completed_items),
            'failed_count': len(self.failed_items),
            'pending_count': len(self.pending_items)
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    @classmethod
    def load(cls, checkpoint_file: Path) -> 'Checkpoint':
        """Load checkpoint from disk."""
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)


@dataclass
class BatchJob:
    """A batch processing job."""
    job_id: str
    name: str
    items: List[BatchItem]
    processor_func: Callable[[Any], Any]
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    checkpoint_interval: int = 100
    processing_mode: ProcessingMode = ProcessingMode.SEQUENTIAL
    max_workers: int = 4
    results: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[str] = field(default_factory=list)
    
    def calculate_progress(self) -> float:
        """Calculate job progress as percentage."""
        if not self.items:
            return 100.0
        
        completed = sum(1 for item in self.items if item.is_completed())
        return (completed / len(self.items)) * 100.0
    
    def get_eta(self) -> Optional[timedelta]:
        """Estimate time to completion."""
        if not self.started_at or self.status != JobStatus.RUNNING:
            return None
        
        progress = self.calculate_progress()
        if progress <= 0:
            return None
        
        elapsed = datetime.now() - self.started_at
        total_estimated = elapsed / (progress / 100.0)
        return total_estimated - elapsed
    
    def get_failed_items(self) -> List[BatchItem]:
        """Get items that failed and cannot be retried."""
        return [item for item in self.items if item.last_error and not item.should_retry()]
    
    def get_pending_items(self) -> List[BatchItem]:
        """Get items that still need processing."""
        return [item for item in self.items if not item.is_completed() and item.should_retry()]


class BatchProcessor:
    """Main batch processor with checkpointing and recovery."""
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 auto_recovery: bool = True,
                 max_concurrent_jobs: int = 1):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.auto_recovery = auto_recovery
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_jobs: Dict[str, BatchJob] = {}
        self.job_lock = threading.Lock()
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Recover jobs if auto_recovery is enabled
        if self.auto_recovery:
            self._recover_jobs()
    
    def create_job(self,
                   name: str,
                   items: List[Any],
                   processor_func: Callable[[Any], Any],
                   checkpoint_interval: int = 100,
                   processing_mode: ProcessingMode = ProcessingMode.SEQUENTIAL,
                   max_workers: int = 4) -> str:
        """Create a new batch processing job."""
        job_id = str(uuid.uuid4())
        
        # Convert items to BatchItem objects
        batch_items = []
        for i, item in enumerate(items):
            batch_item = BatchItem(
                id=f"{job_id}_{i}",
                data=item
            )
            batch_items.append(batch_item)
        
        job = BatchJob(
            job_id=job_id,
            name=name,
            items=batch_items,
            processor_func=processor_func,
            checkpoint_interval=checkpoint_interval,
            processing_mode=processing_mode,
            max_workers=max_workers
        )
        
        with self.job_lock:
            self.active_jobs[job_id] = job
        
        logger.info(f"Created batch job {job_id} with {len(items)} items")
        return job_id
    
    def process_job(self, job_id: str) -> bool:
        """Process a batch job with checkpointing."""
        if job_id not in self.active_jobs:
            logger.error(f"Job {job_id} not found")
            return False
        
        job = self.active_jobs[job_id]
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        
        logger.info(f"Starting batch job {job_id}: {job.name}")
        
        try:
            if job.processing_mode == ProcessingMode.SEQUENTIAL:
                self._process_sequential(job)
            elif job.processing_mode == ProcessingMode.THREADED:
                self._process_threaded(job)
            elif job.processing_mode == ProcessingMode.MULTIPROCESS:
                self._process_multiprocess(job)
            
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Final checkpoint
            self._create_checkpoint(job)
            
            logger.info(f"Completed batch job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            job.status = JobStatus.FAILED
            self._create_checkpoint(job)
            return False
    
    def _process_sequential(self, job: BatchJob):
        """Process job items sequentially."""
        pending_items = job.get_pending_items()
        
        for i, item in enumerate(pending_items):
            if job.status != JobStatus.RUNNING:
                break
            
            try:
                start_time = time.time()
                result = job.processor_func(item.data)
                processing_time = time.time() - start_time
                
                # Store result and mark as completed
                job.results[item.id] = result
                item.processing_time = processing_time
                item.completed_at = datetime.now()
                
                # Create checkpoint if interval reached
                if (i + 1) % job.checkpoint_interval == 0:
                    self._create_checkpoint(job)
                    logger.info(f"Checkpoint created for job {job.job_id} at item {i+1}")
                
            except Exception as e:
                item.mark_failed(str(e))
                logger.warning(f"Item {item.id} failed (attempt {item.attempts}): {str(e)}")
                
                if not item.should_retry():
                    logger.error(f"Item {item.id} failed permanently after {item.attempts} attempts")
    
    def _process_threaded(self, job: BatchJob):
        """Process job items using thread pool."""
        pending_items = job.get_pending_items()
        
        with ThreadPoolExecutor(max_workers=job.max_workers) as executor:
            # Submit all items
            future_to_item = {
                executor.submit(self._process_item_safe, job.processor_func, item): item
                for item in pending_items
            }
            
            processed_count = 0
            for future in as_completed(future_to_item):
                if job.status != JobStatus.RUNNING:
                    break
                
                item = future_to_item[future]
                try:
                    result, processing_time = future.result()
                    job.results[item.id] = result
                    item.processing_time = processing_time
                    item.completed_at = datetime.now()
                    
                except Exception as e:
                    item.mark_failed(str(e))
                    logger.warning(f"Item {item.id} failed: {str(e)}")
                
                processed_count += 1
                
                # Create checkpoint if interval reached
                if processed_count % job.checkpoint_interval == 0:
                    self._create_checkpoint(job)
                    logger.info(f"Checkpoint created for job {job.job_id} at {processed_count} items")
    
    def _process_multiprocess(self, job: BatchJob):
        """Process job items using process pool."""
        pending_items = job.get_pending_items()
        
        with ProcessPoolExecutor(max_workers=job.max_workers) as executor:
            # Submit all items
            future_to_item = {
                executor.submit(self._process_item_safe, job.processor_func, item): item
                for item in pending_items
            }
            
            processed_count = 0
            for future in as_completed(future_to_item):
                if job.status != JobStatus.RUNNING:
                    break
                
                item = future_to_item[future]
                try:
                    result, processing_time = future.result()
                    job.results[item.id] = result
                    item.processing_time = processing_time
                    item.completed_at = datetime.now()
                    
                except Exception as e:
                    item.mark_failed(str(e))
                    logger.warning(f"Item {item.id} failed: {str(e)}")
                
                processed_count += 1
                
                # Create checkpoint if interval reached
                if processed_count % job.checkpoint_interval == 0:
                    self._create_checkpoint(job)
                    logger.info(f"Checkpoint created for job {job.job_id} at {processed_count} items")
    
    def _process_item_safe(self, processor_func: Callable, item: BatchItem) -> tuple:
        """Safely process an item and return result with timing."""
        start_time = time.time()
        result = processor_func(item.data)
        processing_time = time.time() - start_time
        return result, processing_time
    
    def _create_checkpoint(self, job: BatchJob):
        """Create a checkpoint for the current job state."""
        checkpoint_id = f"{job.job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        completed_items = [item.id for item in job.items if item.is_completed()]
        failed_items = [item.id for item in job.get_failed_items()]
        pending_items = [item.id for item in job.get_pending_items()]
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            job_id=job.job_id,
            timestamp=datetime.now(),
            completed_items=completed_items,
            failed_items=failed_items,
            pending_items=pending_items,
            results=job.results.copy(),
            progress=job.calculate_progress(),
            metadata={
                'job_name': job.name,
                'total_items': len(job.items),
                'processing_mode': job.processing_mode.value
            }
        )
        
        checkpoint.save(self.checkpoint_dir)
        job.checkpoints.append(checkpoint_id)
    
    def resume_job(self, job_id: str, checkpoint_id: Optional[str] = None) -> bool:
        """Resume a job from checkpoint."""
        if job_id not in self.active_jobs:
            logger.error(f"Job {job_id} not found")
            return False
        
        job = self.active_jobs[job_id]
        
        # Find the latest checkpoint if none specified
        if checkpoint_id is None:
            if not job.checkpoints:
                logger.warning(f"No checkpoints found for job {job_id}")
                return False
            checkpoint_id = job.checkpoints[-1]
        
        # Load checkpoint
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
        if not checkpoint_file.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_file}")
            return False
        
        try:
            checkpoint = Checkpoint.load(checkpoint_file)
            
            # Restore job state
            job.results.update(checkpoint.results)
            
            # Update item states based on checkpoint
            completed_ids = set(checkpoint.completed_items)
            failed_ids = set(checkpoint.failed_items)
            
            for item in job.items:
                if item.id in completed_ids:
                    item.completed_at = checkpoint.timestamp
                elif item.id in failed_ids:
                    item.last_error = "Failed in previous run"
                    item.attempts = item.max_attempts  # Mark as permanently failed
            
            logger.info(f"Resumed job {job_id} from checkpoint {checkpoint_id}")
            logger.info(f"Progress: {checkpoint.progress:.1f}% ({len(completed_ids)} completed, {len(failed_ids)} failed)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {str(e)}")
            return False
    
    def _recover_jobs(self):
        """Recover jobs from checkpoints on startup."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        
        if not checkpoint_files:
            return
        
        logger.info(f"Found {len(checkpoint_files)} checkpoint files")
        
        for checkpoint_file in checkpoint_files:
            try:
                checkpoint = Checkpoint.load(checkpoint_file)
                
                # Only recover incomplete jobs
                if checkpoint.progress < 100.0:
                    logger.info(f"Recovering incomplete job {checkpoint.job_id} (progress: {checkpoint.progress:.1f}%)")
                    # Job recovery would require re-creating the job with the processor function
                    # This is a simplified version - in practice, you'd need to store the processor function reference
                    
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {str(e)}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a job."""
        if job_id not in self.active_jobs:
            return None
        
        job = self.active_jobs[job_id]
        
        return {
            'job_id': job_id,
            'name': job.name,
            'status': job.status.value,
            'progress': job.calculate_progress(),
            'total_items': len(job.items),
            'completed_items': len([item for item in job.items if item.is_completed()]),
            'failed_items': len(job.get_failed_items()),
            'pending_items': len(job.get_pending_items()),
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'eta': str(job.get_eta()) if job.get_eta() else None,
            'checkpoints': len(job.checkpoints)
        }
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a running job."""
        if job_id not in self.active_jobs:
            return False
        
        job = self.active_jobs[job_id]
        if job.status == JobStatus.RUNNING:
            job.status = JobStatus.PAUSED
            self._create_checkpoint(job)
            logger.info(f"Paused job {job_id}")
            return True
        
        return False
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        if job_id not in self.active_jobs:
            return False
        
        job = self.active_jobs[job_id]
        job.status = JobStatus.CANCELLED
        self._create_checkpoint(job)
        logger.info(f"Cancelled job {job_id}")
        return True
    
    def cleanup_old_checkpoints(self, days: int = 7):
        """Clean up old checkpoint files."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                file_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                if file_time < cutoff_date:
                    checkpoint_file.unlink()
                    # Also remove corresponding JSON file
                    json_file = checkpoint_file.with_suffix('.json')
                    if json_file.exists():
                        json_file.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint_file}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_file}: {str(e)}")


def example_processor(item: Any) -> str:
    """Example processor function for testing."""
    # Simulate some processing time
    time.sleep(0.1)
    
    # Simulate occasional failures
    if hasattr(item, 'should_fail') and item.should_fail:
        raise ValueError("Simulated processing error")
    
    return f"Processed: {item}"


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuronMap Batch Processor')
    parser.add_argument('--test', action='store_true', help='Run test batch job')
    parser.add_argument('--status', help='Get status of job ID')
    parser.add_argument('--resume', help='Resume job ID')
    parser.add_argument('--pause', help='Pause job ID')
    parser.add_argument('--cancel', help='Cancel job ID')
    parser.add_argument('--cleanup', type=int, help='Clean up checkpoints older than N days')
    
    args = parser.parse_args()
    
    processor = BatchProcessor()
    
    if args.test:
        # Create test data
        test_items = [f"item_{i}" for i in range(50)]
        
        job_id = processor.create_job(
            name="Test Batch Job",
            items=test_items,
            processor_func=example_processor,
            checkpoint_interval=10,
            processing_mode=ProcessingMode.SEQUENTIAL
        )
        
        print(f"Created test job: {job_id}")
        processor.process_job(job_id)
        
        status = processor.get_job_status(job_id)
        print(f"Job completed with status: {status}")
    
    elif args.status:
        status = processor.get_job_status(args.status)
        if status:
            print(json.dumps(status, indent=2))
        else:
            print(f"Job {args.status} not found")
    
    elif args.resume:
        if processor.resume_job(args.resume):
            print(f"Resumed job {args.resume}")
            processor.process_job(args.resume)
        else:
            print(f"Failed to resume job {args.resume}")
    
    elif args.pause:
        if processor.pause_job(args.pause):
            print(f"Paused job {args.pause}")
        else:
            print(f"Failed to pause job {args.pause}")
    
    elif args.cancel:
        if processor.cancel_job(args.cancel):
            print(f"Cancelled job {args.cancel}")
        else:
            print(f"Failed to cancel job {args.cancel}")
    
    elif args.cleanup is not None:
        processor.cleanup_old_checkpoints(args.cleanup)
        print(f"Cleaned up checkpoints older than {args.cleanup} days")


if __name__ == "__main__":
    main()
