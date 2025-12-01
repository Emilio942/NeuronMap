"""
Test Redis Queue Integration
===========================

This script tests the Redis Task Queue integration by:
1. Starting a worker process in the background.
2. Submitting a task via the Orchestrator using Redis.
3. Waiting for the result.
4. Verifying the output.

Requirements:
- Redis server must be running locally on default port (6379).
- `redis` python package must be installed.
"""

import sys
import os
import time
import threading
import logging
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath("."))

# Mock Redis for testing in environment without redis-server
import fakeredis
import redis

# Patch redis.Redis to use FakeRedis
# We need to do this BEFORE importing modules that use redis
redis.Redis = fakeredis.FakeRedis

from src.core.orchestrator import SystemOrchestrator
from src.core.task_queue import TaskStatus
from src.worker import Worker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_redis")

def run_worker_thread(stop_event):
    """Run the worker in a thread."""
    worker = Worker()
    # We need to make the worker loop check the stop event
    # But the worker.start() is an infinite loop with blocking pop.
    # We can't easily stop it if it's blocking on blpop with fakeredis?
    # fakeredis supports blpop.
    
    logger.info("Worker thread started")
    while not stop_event.is_set():
        try:
            # Use a short timeout to allow checking stop_event
            item = worker.queue.redis.blpop(worker.queue.queue_name, timeout=1)
            if item:
                _, data = item
                worker.process_task(data)
        except Exception as e:
            logger.error(f"Worker thread error: {e}")
            time.sleep(0.1)

def test_redis_queue():
    logger.info("Testing with FakeRedis...")

    # 1. Start Worker Thread
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=run_worker_thread, args=(stop_event,))
    worker_thread.daemon = True
    worker_thread.start()
    
    try:
        # 2. Initialize Orchestrator with Redis queue
        # Use a temporary directory for this test
        test_dir = "test_redis_env"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            
        # Note: SystemOrchestrator will create a RedisTaskQueue.
        # Since we patched redis.Redis, it will use FakeRedis.
        # And since FakeRedis is in-memory and we are in the same process,
        # they share the data!
        orchestrator = SystemOrchestrator(base_dir=test_dir, queue_type="redis")
        
        # 3. Create Project
        project_id = orchestrator.create_project("Redis Test Project", "Testing distributed queue")
        logger.info(f"Created project: {project_id}")
        
        # 4. Submit Analysis
        input_data = "Test Input Data"
        model_name = "gpt2" 
        
        logger.info("Submitting analysis task...")
        experiment_id = orchestrator.submit_analysis_pipeline(
            project_id,
            model_name,
            input_data,
            ["statistical"]
        )
        logger.info(f"Task submitted. Experiment ID: {experiment_id}")
        
        # 5. Poll for completion
        logger.info("Waiting for results...")
        max_retries = 10
        for i in range(max_retries):
            experiment = orchestrator.project_manager.get_experiment(project_id, experiment_id)
            status = experiment.get("status")
            logger.info(f"Status check {i+1}/{max_retries}: {status}")
            
            if status == "completed":
                logger.info("Task completed successfully!")
                results = experiment.get("results", {})
                logger.info(f"Results keys: {list(results.keys())}")
                break
            elif status == "failed":
                logger.error(f"Task failed: {experiment.get('error')}")
                break
            
            time.sleep(1)
        else:
            logger.error("Timeout waiting for task completion.")

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
    finally:
        # 6. Cleanup
        logger.info("Stopping worker...")
        stop_event.set()
        worker_thread.join(timeout=2)
        
        # Cleanup test dir
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        logger.info("Test finished.")

if __name__ == "__main__":
    test_redis_queue()
