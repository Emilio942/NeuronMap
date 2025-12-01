"""
NeuronMap Worker
===============

Background worker process that consumes tasks from Redis.
"""

import sys
import os
import time
import logging
import pickle
import importlib
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath("."))

from src.core.task_queue import RedisTaskQueue, TaskStatus
from src.core.orchestrator import SystemOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("worker")

class Worker:
    def __init__(self, host='localhost', port=6379, db=0, queue_name='neuronmap_tasks'):
        self.queue = RedisTaskQueue(host, port, db, queue_name)
        self.orchestrator = SystemOrchestrator() # Initialize orchestrator for context
        logger.info(f"Worker initialized, listening on {queue_name}")

    def start(self):
        logger.info("Worker started. Waiting for tasks...")
        while True:
            try:
                # Blocking pop from Redis
                # Returns (queue_name, data)
                item = self.queue.redis.blpop(self.queue.queue_name, timeout=5)
                
                if item:
                    _, data = item
                    self.process_task(data)
                
            except KeyboardInterrupt:
                logger.info("Worker stopping...")
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(1)

    def process_task(self, data: bytes):
        try:
            payload = pickle.loads(data)
            task_id = payload["task_id"]
            func_name = payload["func_name"]
            module_name = payload["module"]
            args = payload["args"]
            kwargs = payload["kwargs"]
            
            logger.info(f"Processing task {task_id}: {module_name}.{func_name}")
            
            # Update status to RUNNING
            self.queue._update_status(task_id, TaskStatus.RUNNING)
            
            # Dynamic import and execution
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            
            # Execute
            result = func(*args, **kwargs)
            
            # Update status to COMPLETED
            self.queue._update_status(task_id, TaskStatus.COMPLETED, result=result)
            logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            # Log full traceback for debugging
            logger.error(traceback.format_exc())
            if 'task_id' in locals():
                self.queue._update_status(task_id, TaskStatus.FAILED, error=str(e))

# Standalone task wrapper to avoid pickling the Orchestrator instance
def run_analysis_task(project_id, experiment_id, model_name, input_data, analysis_types):
    """
    Standalone function that instantiates an Orchestrator and runs the analysis.
    This function is picklable.
    """
    orchestrator = SystemOrchestrator()
    return orchestrator._run_analysis_worker(
        project_id, experiment_id, model_name, input_data, analysis_types
    )

if __name__ == "__main__":
    worker = Worker()
    worker.start()
if __name__ == "__main__":
    worker = Worker()
    worker.start()