"""
Job queue system for CondaEnvExecutor.

This module provides a job queue for asynchronous execution of Python code in 
isolated conda environments, with status tracking and result retrieval.
"""

import asyncio
import time
import uuid
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import threading
from enum import Enum

from .executor import ExecutionResult, TimingInfo, CondaEnvExecutor


class JobStatus(Enum):
    """Enum for job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Information about a job in the queue."""
    job_id: str
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    user_id: Optional[str] = None
    dependencies: Optional[List[Union[str, Dict]]] = None
    channels: Optional[List[str]] = None
    error: Optional[str] = None


@dataclass
class JobResult:
    """Result of a completed job."""
    job_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    stdout: str = ''
    stderr: str = ''
    timing: Optional[Dict] = None
    
    @classmethod
    def from_execution_result(cls, job_id: str, result: ExecutionResult) -> 'JobResult':
        """Create JobResult from ExecutionResult."""
        return cls(
            job_id=job_id,
            success=result.success,
            result=result.result,
            error=result.error,
            stdout=result.stdout,
            stderr=result.stderr,
            timing=asdict(result.timing) if result.timing else None
        )


class JobQueue:
    """A queue for managing asynchronous code execution jobs."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the job queue.
        
        Args:
            storage_dir: Directory to store job results. Defaults to ~/.conda_env_jobs
        """
        self.storage_dir = Path(storage_dir or os.path.expanduser("~/.conda_env_jobs"))
        self.jobs_dir = self.storage_dir / "jobs"
        self.results_dir = self.storage_dir / "results"
        
        # Create directories if they don't exist
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory tracking of jobs
        self.jobs: Dict[str, JobInfo] = {}
        self._load_existing_jobs()
        
        # Queue for pending jobs
        self.pending_queue: asyncio.Queue = asyncio.Queue()
        
        # Lock for job operations
        self._lock = threading.RLock()
        
        # Flag to control worker
        self._worker_running = False
        self._worker_task = None
    
    def _load_existing_jobs(self) -> None:
        """Load existing jobs from storage."""
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file, "r") as f:
                    job_data = json.load(f)
                job_id = job_data.pop("job_id")
                status = JobStatus(job_data.pop("status"))
                job_info = JobInfo(job_id=job_id, status=status, **job_data)
                self.jobs[job_id] = job_info
            except Exception as e:
                print(f"Error loading job {job_file}: {e}")
    
    def _save_job_info(self, job_info: JobInfo) -> None:
        """Save job information to storage."""
        job_path = self.jobs_dir / f"{job_info.job_id}.json"
        with open(job_path, "w") as f:
            job_dict = asdict(job_info)
            job_dict["status"] = job_info.status.value  # Convert enum to string
            json.dump(job_dict, f)
    
    def _save_job_result(self, job_result: JobResult) -> None:
        """Save job result to storage."""
        result_path = self.results_dir / f"{job_result.job_id}.json"
        with open(result_path, "w") as f:
            json.dump(asdict(job_result), f)
    
    async def submit_job(
        self, 
        code: str, 
        input_data: Any = None,
        dependencies: List[Union[str, Dict]] = None,
        channels: List[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Submit a new job to the queue.
        
        Args:
            code: The Python code string to execute.
            input_data: Data to be passed to the `execute` function.
            dependencies: List of conda/pip packages to install.
            channels: List of Conda channels to use.
            user_id: Optional user identifier for tracking.
            
        Returns:
            job_id: The ID of the submitted job.
        """
        job_id = str(uuid.uuid4())
        
        with self._lock:
            # Create job info
            job_info = JobInfo(
                job_id=job_id,
                status=JobStatus.PENDING,
                created_at=time.time(),
                user_id=user_id,
                dependencies=dependencies,
                channels=channels
            )
            
            # Save to memory and disk
            self.jobs[job_id] = job_info
            self._save_job_info(job_info)
            
            # Add to queue
            await self.pending_queue.put({
                "job_id": job_id,
                "code": code,
                "input_data": input_data,
                "dependencies": dependencies,
                "channels": channels,
                "user_id": user_id
            })
            
            # Start worker if not running
            if not self._worker_running:
                await self.start_worker()
                
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """
        Get the status of a job.
        
        Args:
            job_id: The ID of the job.
            
        Returns:
            A dictionary with job status information or None if job not found.
        """
        with self._lock:
            job_info = self.jobs.get(job_id)
            if job_info:
                job_dict = asdict(job_info)
                job_dict["status"] = job_info.status.value  # Convert enum to string
                return job_dict
            return None
    
    async def get_job_result(self, job_id: str) -> Optional[Dict]:
        """
        Get the result of a completed job.
        
        Args:
            job_id: The ID of the job.
            
        Returns:
            A dictionary with job result or None if not available.
        """
        result_path = self.results_dir / f"{job_id}.json"
        if result_path.exists():
            try:
                with open(result_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading result for job {job_id}: {e}")
                return None
        return None
    
    async def wait_for_result(self, job_id: str, timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Wait for a job to complete and return its result.
        
        Args:
            job_id: The ID of the job.
            timeout: Maximum time to wait in seconds.
            
        Returns:
            A dictionary with job result or None if timeout reached.
        """
        start_time = time.time()
        while timeout is None or (time.time() - start_time) < timeout:
            with self._lock:
                job_info = self.jobs.get(job_id)
                if not job_info:
                    return None
                
                if job_info.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    return await self.get_job_result(job_id)
            
            await asyncio.sleep(0.5)
        
        return None
    
    async def start_worker(self):
        """Start the worker task to process jobs."""
        if self._worker_running:
            return
        
        self._worker_running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
    
    async def stop_worker(self):
        """Stop the worker task."""
        self._worker_running = False
        if self._worker_task:
            try:
                self._worker_task.cancel()
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
    
    async def _worker_loop(self):
        """Main worker loop for processing jobs."""
        loop = asyncio.get_running_loop()
        
        while self._worker_running:
            try:
                # Get next job from queue
                job_data = await self.pending_queue.get()
                job_id = job_data["job_id"]
                
                # Check if job has been canceled before processing
                with self._lock:
                    job_info = self.jobs.get(job_id)
                    # If job no longer exists or is already marked as failed, skip it
                    if not job_info or job_info.status == JobStatus.FAILED:
                        print(f"Job {job_id} was canceled before execution started. Skipping.")
                        self.pending_queue.task_done()
                        continue
                        
                    # Mark job as running
                    job_info.status = JobStatus.RUNNING
                    job_info.started_at = time.time()
                    self._save_job_info(job_info)
                
                # Execute the job
                executor = None
                try:
                    executor = await loop.run_in_executor(
                        None,
                        CondaEnvExecutor.create_temp_env,
                        job_data["dependencies"],
                        job_data["channels"]
                    )
                    
                    # Check if job was canceled during environment setup
                    with self._lock:
                        job_info = self.jobs.get(job_id)
                        if not job_info or job_info.status == JobStatus.FAILED:
                            print(f"Job {job_id} was canceled during environment setup. Skipping execution.")
                            # Job was canceled, cleanup and continue
                            raise asyncio.CancelledError()
                    
                    result = await loop.run_in_executor(
                        None,
                        executor.execute,
                        job_data["code"],
                        job_data["input_data"]
                    )
                    
                    # Check if job was canceled during execution
                    with self._lock:
                        job_info = self.jobs.get(job_id)
                        if not job_info or job_info.status == JobStatus.FAILED:
                            print(f"Job {job_id} was canceled during execution. Discarding results.")
                            # Job was canceled, we'll honor the cancel status rather than updating with results
                            raise asyncio.CancelledError()
                    
                    # Create job result
                    job_result = JobResult.from_execution_result(job_id, result)
                    
                    # Update job status
                    with self._lock:
                        job_info = self.jobs[job_id]
                        job_info.status = JobStatus.COMPLETED if result.success else JobStatus.FAILED
                        job_info.completed_at = time.time()
                        job_info.error = result.error
                        self._save_job_info(job_info)
                    
                    # Save result
                    self._save_job_result(job_result)
                    
                except asyncio.CancelledError:
                    # Job was canceled, any necessary cleanup should have been done
                    # when the job was marked as canceled
                    pass
                    
                except Exception as e:
                    # Handle execution error
                    with self._lock:
                        job_info = self.jobs.get(job_id)
                        if job_info and job_info.status != JobStatus.FAILED:
                            # Only update if job wasn't canceled
                            job_info.status = JobStatus.FAILED
                            job_info.completed_at = time.time()
                            job_info.error = str(e)
                            self._save_job_info(job_info)
                    
                        error_result = JobResult(
                            job_id=job_id,
                            success=False,
                            error=f"Execution failed: {str(e)}",
                            stdout="",
                            stderr=f"Exception during execution: {str(e)}"
                        )
                        self._save_job_result(error_result)
                
                finally:
                    # Clean up executor
                    if executor:
                        await loop.run_in_executor(None, executor.cleanup)
                    
                    # Mark task as done
                    self.pending_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in worker loop: {e}")
                await asyncio.sleep(1)  # Avoid tight loop on repeated errors

    async def list_jobs(self, user_id: Optional[str] = None, limit: int = 100, status: Optional[str] = None) -> List[Dict]:
        """
        List jobs, optionally filtered by user and status.
        
        Args:
            user_id: Filter by user ID.
            limit: Maximum number of jobs to return.
            status: Filter by job status.
            
        Returns:
            List of job information dictionaries.
        """
        with self._lock:
            jobs = list(self.jobs.values())
            
            # Apply filters
            if user_id:
                jobs = [job for job in jobs if job.user_id == user_id]
            
            if status:
                try:
                    status_enum = JobStatus(status)
                    jobs = [job for job in jobs if job.status == status_enum]
                except ValueError:
                    pass  # Invalid status, ignore filter
            
            # Sort by created_at (newest first) and limit
            jobs.sort(key=lambda job: job.created_at, reverse=True)
            jobs = jobs[:limit]
            
            # Convert to dictionaries
            result = []
            for job in jobs:
                job_dict = asdict(job)
                job_dict["status"] = job.status.value  # Convert enum to string
                result.append(job_dict)
            
            return result
            
    async def cancel_job(self, job_id: str, user_id: Optional[str] = None) -> Dict:
        """
        Cancel a job if it's still in the queue or running.
        
        Args:
            job_id: The ID of the job to cancel.
            user_id: The ID of the user trying to cancel the job.
                     If provided, will verify job ownership.
            
        Returns:
            Dictionary with cancel result:
            {
                "success": bool,
                "job_id": str,
                "message": str,
                "status": str
            }
        """
        with self._lock:
            job_info = self.jobs.get(job_id)
            
            if not job_info:
                return {
                    "success": False,
                    "job_id": job_id,
                    "message": "Job not found",
                    "status": None
                }
            
            # Verify job ownership if user_id is provided
            if user_id and job_info.user_id and job_info.user_id != user_id:
                return {
                    "success": False,
                    "job_id": job_id,
                    "message": "Permission denied: job belongs to another user",
                    "status": job_info.status.value
                }
            
            # Check if job is already completed or failed
            if job_info.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                return {
                    "success": False,
                    "job_id": job_id,
                    "message": f"Cannot cancel job with status: {job_info.status.value}",
                    "status": job_info.status.value
                }
            
            # If job is pending, we can simply update its status
            if job_info.status == JobStatus.PENDING:
                job_info.status = JobStatus.CANCELLED
                job_info.completed_at = time.time()
                job_info.error = "Job cancelled by user"
                self._save_job_info(job_info)
                
                # Create a cancelled result
                cancel_result = JobResult(
                    job_id=job_id,
                    success=False,
                    error="Job cancelled by user",
                    stdout="",
                    stderr="Job was cancelled before execution"
                )
                self._save_job_result(cancel_result)
                
                return {
                    "success": True,
                    "job_id": job_id,
                    "message": "Job cancelled successfully",
                    "status": job_info.status.value
                }
            
            # If job is running, we need to mark it for cancellation
            # The worker will check this flag and terminate the job
            if job_info.status == JobStatus.RUNNING:
                # Mark for cancellation - the actual cancellation will happen
                # in the worker loop when it checks this flag
                job_info.status = JobStatus.CANCELLED
                job_info.completed_at = time.time()
                job_info.error = "Job cancelled by user"
                self._save_job_info(job_info)
                
                # Create a cancelled result
                cancel_result = JobResult(
                    job_id=job_id,
                    success=False,
                    error="Job cancelled by user",
                    stdout="",
                    stderr="Job was cancelled during execution"
                )
                self._save_job_result(cancel_result)
                
                return {
                    "success": True,
                    "job_id": job_id,
                    "message": "Job marked for cancellation",
                    "status": job_info.status.value
                }
            
            # Should not reach here
            return {
                "success": False,
                "job_id": job_id,
                "message": "Unable to cancel job: unknown status",
                "status": job_info.status.value
            } 