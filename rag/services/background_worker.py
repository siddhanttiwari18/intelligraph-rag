import time
import uuid
import logging
import threading
from typing import Callable, Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("rag_platform")

class BackgroundTaskManager:
    """Manages execution and status tracking of asynchronous tasks like PDF parsing,
    OCR, and graph rebuilding.
    """
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit_task(self, fn: Callable, *args, description: str = "", **kwargs) -> str:
        """Submit a task to be run in the background.
        
        Returns the unique task_id.
        """
        task_id = str(uuid.uuid4())
        
        with self._lock:
            self._tasks[task_id] = {
                "id": task_id,
                "description": description,
                "status": "PENDING",
                "progress": 0.0,
                "message": "Waiting for execution queue...",
                "start_time": time.time(),
                "end_time": None,
                "result": None,
                "error": None
            }

        def wrapper():
            self._update_task_status(task_id, "RUNNING", progress=0.1, message="Started execution...")
            try:
                # Execute the actual function
                logger.info(f"Starting background task {task_id} ({description})")
                result = fn(*args, **kwargs)
                self._update_task_status(task_id, "COMPLETED", progress=1.0, message="Success", result=result)
                logger.info(f"Completed background task {task_id} successfully")
            except Exception as e:
                logger.exception(f"Error in background task {task_id}")
                self._update_task_status(task_id, "FAILED", progress=1.0, message="Failed", error=str(e))

        self._executor.submit(wrapper)
        return task_id

    def _update_task_status(
        self, 
        task_id: str, 
        status: str, 
        progress: float, 
        message: str, 
        result: Any = None, 
        error: str = None
    ) -> None:
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task["status"] = status
                task["progress"] = progress
                task["message"] = message
                if result is not None:
                    task["result"] = result
                if error is not None:
                    task["error"] = error
                if status in ("COMPLETED", "FAILED"):
                    task["end_time"] = time.time()

    def get_task_status(self, task_id: str) -> Optional[dict]:
        """Get the current status of a specific task."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, dict]:
        """Get all tasks currently registered."""
        with self._lock:
            return dict(self._tasks)

    def remove_task(self, task_id: str) -> None:
        """Remove a task from history."""
        with self._lock:
            self._tasks.pop(task_id, None)

# Singleton manager
background_worker = BackgroundTaskManager()
