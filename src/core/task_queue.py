"""
Task Queue System
================

Manages background execution of long-running analysis tasks.
Supports local execution (threading) and extensible for distributed queues (Redis/Celery).
"""

import logging
import uuid
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum

import pickle
import json
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskResult:
    def __init__(self, task_id: str, status: TaskStatus, result: Any = None, error: str = None):
        self.task_id = task_id
        self.status = status
        self.result = result
        self.error = error
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskResult':
        status = TaskStatus(data["status"])
        obj = cls(data["task_id"], status, data.get("result"), data.get("error"))
        if "updated_at" in data:
            obj.updated_at = datetime.fromisoformat(data["updated_at"])
        return obj

class TaskQueueBase(ABC):
    """Abstract base class for task queues."""
    
    @abstractmethod
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task for execution. Returns task_id."""
        pass

    @abstractmethod
    def get_status(self, task_id: str) -> Optional[TaskResult]:
        """Get the status of a task."""
        pass

class RedisTaskQueue(TaskQueueBase):
    """
    Distributed task queue using Redis.
    Requires a separate worker process.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, queue_name: str = 'neuronmap_tasks', **kwargs):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not installed. Install with: pip install redis")
            
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self.queue_name = queue_name
        self.status_prefix = "task_status:"
        
        try:
            self.redis.ping()
            logger.info(f"RedisTaskQueue connected to {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            raise

    def submit(self, func: Callable, *args, **kwargs) -> str:
        task_id = str(uuid.uuid4())
        
        # Prepare task payload
        # We pickle the function and args. 
        # Note: func must be importable by the worker!
        payload = {
            "task_id": task_id,
            "func_name": func.__name__,
            "module": func.__module__,
            "args": args,
            "kwargs": kwargs,
            "submitted_at": datetime.now().isoformat()
        }
        
        # Initialize status in Redis
        self._update_status(task_id, TaskStatus.PENDING)
        
        # Push to queue
        self.redis.rpush(self.queue_name, pickle.dumps(payload))
        
        logger.info(f"Task submitted to Redis: {task_id}")
        return task_id

    def get_status(self, task_id: str) -> Optional[TaskResult]:
        data = self.redis.get(f"{self.status_prefix}{task_id}")
        if data:
            return TaskResult.from_dict(pickle.loads(data))
        return None

    def _update_status(self, task_id: str, status: TaskStatus, result: Any = None, error: str = None):
        task_result = TaskResult(task_id, status, result, error)
        # Expire status after 24 hours
        self.redis.setex(
            f"{self.status_prefix}{task_id}", 
            86400, 
            pickle.dumps(task_result.to_dict())
        )

class LocalTaskQueue(TaskQueueBase):
    """
    Local task queue using ThreadPoolExecutor.
    Suitable for development and single-instance deployments.
    """
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, TaskResult] = {}
        self.futures: Dict[str, Future] = {}
        logger.info(f"LocalTaskQueue initialized with {max_workers} workers")

    def submit(self, func: Callable, *args, **kwargs) -> str:
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        self.tasks[task_id] = TaskResult(task_id, TaskStatus.PENDING)
        
        # Submit to executor
        # We wrap the function to handle status updates
        future = self.executor.submit(self._worker_wrapper, task_id, func, *args, **kwargs)
        self.futures[task_id] = future
        
        logger.info(f"Task submitted: {task_id}")
        return task_id

    def _worker_wrapper(self, task_id: str, func: Callable, *args, **kwargs):
        """Internal wrapper to handle status updates and errors."""
        try:
            logger.info(f"Task started: {task_id}")
            self._update_status(task_id, TaskStatus.RUNNING)
            
            # Execute the actual function
            result = func(*args, **kwargs)
            
            self._update_status(task_id, TaskStatus.COMPLETED, result=result)
            logger.info(f"Task completed: {task_id}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Task failed {task_id}: {error_msg}\n{stack_trace}")
            self._update_status(task_id, TaskStatus.FAILED, error=error_msg)
            raise

    def _update_status(self, task_id: str, status: TaskStatus, result: Any = None, error: str = None):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            task.updated_at = datetime.now()

    def get_status(self, task_id: str) -> Optional[TaskResult]:
        return self.tasks.get(task_id)

# Factory for creating task queues
def create_task_queue(queue_type: str = "local", **kwargs) -> TaskQueueBase:
    if queue_type == "local":
        return LocalTaskQueue(**kwargs)
    elif queue_type == "redis":
        return RedisTaskQueue(**kwargs)
    else:
        raise ValueError(f"Unknown queue type: {queue_type}")
