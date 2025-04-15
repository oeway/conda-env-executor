import asyncio
import argparse
import os
import traceback
import uuid
from typing import Union, Dict, Optional, Any, List
import sys
from hypha_rpc import login, connect_to_server

# Ensure the executor module can be found, especially when running as a script
try:
    from conda_env_executor.executor import CondaEnvExecutor, ExecutionResult, TimingInfo
    from conda_env_executor.job_queue import JobQueue
except ImportError:
    # Add the parent directory to sys.path to find the package
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PARENT_DIR = os.path.dirname(_CURRENT_DIR)
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)
    from conda_env_executor.executor import CondaEnvExecutor, ExecutionResult, TimingInfo
    from conda_env_executor.job_queue import JobQueue

# Create a global job queue instance
job_queue = JobQueue()


async def execute_in_conda_env(
    code: str,
    input_data: any = None,
    dependencies: list[Union[str, Dict]] = None,
    channels: list[str] = None,
    context: any = None,
) -> dict:
    """
    Executes Python code within a dynamically created Conda environment.

    Args:
        code: The Python code string. Must contain a function `execute(input_data)`.
        input_data: Data to be passed to the `execute` function. Must be serializable.
        dependencies: List of conda/pip packages to install in the environment.
                     If None, defaults to ["python"].
        channels: List of Conda channels to use. Defaults to ["conda-forge"].

    Returns:
        A dictionary containing execution results:
        {
            "success": bool,
            "result": any, # Output of the execute function
            "error": str | None,
            "stdout": str | None,
            "stderr": str | None,
            "timing": dict | None # Env creation and execution times
        }
    """
    print(f"User {context['user']['id']} is executing code...")
    if not code:
        return {"success": False, "error": "No code provided."}

    if dependencies is None:
        # Default to just python if dependencies list is empty or None
        dependencies = ["python"]
        print("Warning: No dependencies specified. Using default: ['python']")

    if channels is None:
        channels = ["conda-forge"]

    executor = None
    loop = asyncio.get_running_loop()

    try:
        print(f"Attempting to set up environment with dependencies: {dependencies}")

        # --- Create Environment ---
        executor = await loop.run_in_executor(
            None,
            CondaEnvExecutor.create_temp_env,
            dependencies,
            channels
        )

        print(f"Environment ready at: {executor.env_path}")

        # --- Execute Code ---
        print("Executing user code...")
        result: ExecutionResult = await loop.run_in_executor(
            None, # Use default ThreadPoolExecutor
            executor.execute,
            code,
            input_data
        )
        print(f"Execution finished. Success: {result.success}")

        # --- Format Result ---
        return {
            "success": result.success,
            "result": result.result, # Ensure result is JSON serializable by the caller/Hypha
            "error": result.error,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timing": {
                "env_setup_time": result.timing.env_setup_time,
                "execution_time": result.timing.execution_time,
                "total_time": result.timing.total_time,
            },
        }

    except Exception as e:
        print(f"Error during service execution: {e}")
        return {
            "success": False,
            "error": f"Service execution failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }
    finally:
        # --- Cleanup ---
        if executor:
            print(f"Cleaning up environment: {executor.env_path}")
            try:
                await loop.run_in_executor(None, executor.cleanup)
                print("Cleanup successful.")
            except Exception as e:
                 print(f"Error during executor cleanup: {e}")
                 # Log error but don't prevent service from returning response


async def submit_job(
    code: str,
    input_data: Any = None,
    dependencies: List[Union[str, Dict]] = None,
    channels: List[str] = None,
    context: Any = None,
) -> dict:
    """
    Submits a job for asynchronous execution in a conda environment.
    
    Args:
        code: The Python code string to execute.
        input_data: Data to be passed to the execute function.
        dependencies: List of conda/pip packages to install.
        channels: List of conda channels to use.
        context: Execution context with user information.
        
    Returns:
        Dictionary with job information:
        {
            "job_id": str,
            "status": str,
            "created_at": float
        }
    """
    print(f"User {context['user']['id']} is submitting a job...")
    
    if not code:
        return {"success": False, "error": "No code provided."}
    
    if dependencies is None:
        dependencies = ["python"]
        print("Warning: No dependencies specified. Using default: ['python']")
    
    if channels is None:
        channels = ["conda-forge"]
    
    try:
        # Submit job to queue
        user_id = context['user']['id'] if context and 'user' in context else None
        job_id = await job_queue.submit_job(
            code=code,
            input_data=input_data,
            dependencies=dependencies,
            channels=channels,
            user_id=user_id
        )
        
        # Get job status
        job_status = await job_queue.get_job_status(job_id)
        
        return {
            "success": True,
            "job_id": job_id,
            "status": job_status
        }
        
    except Exception as e:
        print(f"Error submitting job: {e}")
        return {
            "success": False,
            "error": f"Job submission failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


async def get_job_status(
    job_id: str,
    context: Any = None
) -> dict:
    """
    Retrieves the status of a job.
    
    Args:
        job_id: ID of the job to check.
        context: Execution context with user information.
    
    Returns:
        Dictionary with job status:
        {
            "success": bool,
            "job_id": str,
            "status": str,
            "created_at": float,
            ...
        }
    """
    print(f"User {context['user']['id']} is checking job status for: {job_id}")
    
    try:
        job_status = await job_queue.get_job_status(job_id)
        
        if job_status:
            return {
                "success": True,
                "job_id": job_id,
                "status": job_status
            }
        else:
            return {
                "success": False,
                "error": f"Job {job_id} not found"
            }
            
    except Exception as e:
        print(f"Error retrieving job status: {e}")
        return {
            "success": False,
            "error": f"Failed to get job status: {str(e)}"
        }


async def get_job_result(
    job_id: str,
    context: Any = None
) -> dict:
    """
    Retrieves the result of a completed job.
    
    Args:
        job_id: ID of the job.
        context: Execution context with user information.
    
    Returns:
        Dictionary with job result if completed:
        {
            "success": bool,
            "job_id": str,
            "result": any,
            "error": str | None,
            "stdout": str,
            "stderr": str,
            "timing": dict | None
        }
    """
    print(f"User {context['user']['id']} is retrieving result for job: {job_id}")
    
    try:
        # First check if job exists
        job_status = await job_queue.get_job_status(job_id)
        
        if not job_status:
            return {
                "success": False,
                "error": f"Job {job_id} not found"
            }
        
        # Check if job is completed
        status = job_status.get("status")
        if status not in ("completed", "failed"):
            return {
                "success": False,
                "error": f"Job {job_id} is not completed yet (status: {status})",
                "status": job_status
            }
        
        # Get the result
        result = await job_queue.get_job_result(job_id)
        
        if result:
            return {
                "success": True,
                "job_id": job_id,
                "result": result
            }
        else:
            return {
                "success": False,
                "error": f"Result for job {job_id} not found"
            }
            
    except Exception as e:
        print(f"Error retrieving job result: {e}")
        return {
            "success": False,
            "error": f"Failed to get job result: {str(e)}"
        }


async def wait_for_result(
    job_id: str,
    timeout: Optional[float] = None,
    context: Any = None
) -> dict:
    """
    Waits for a job to complete and returns its result.
    
    Args:
        job_id: ID of the job.
        timeout: Maximum time to wait in seconds.
        context: Execution context with user information.
    
    Returns:
        Dictionary with job result:
        {
            "success": bool,
            "job_id": str,
            "result": any,
            ...
        }
    """
    print(f"User {context['user']['id']} is waiting for result of job: {job_id}")
    
    try:
        # First check if job exists
        job_status = await job_queue.get_job_status(job_id)
        
        if not job_status:
            return {
                "success": False,
                "error": f"Job {job_id} not found"
            }
        
        # Wait for the result
        result = await job_queue.wait_for_result(job_id, timeout)
        
        if result:
            return {
                "success": True,
                "job_id": job_id,
                "result": result
            }
        else:
            # If we get here, it's likely due to a timeout
            current_status = await job_queue.get_job_status(job_id)
            return {
                "success": False,
                "error": "Timeout waiting for job completion" if timeout else "Failed to get job result",
                "status": current_status
            }
            
    except Exception as e:
        print(f"Error waiting for job result: {e}")
        return {
            "success": False,
            "error": f"Failed to wait for job result: {str(e)}"
        }


async def list_jobs(
    user_id: Optional[str] = None,
    limit: int = 100,
    status: Optional[str] = None,
    context: Any = None
) -> dict:
    """
    Lists jobs, optionally filtered by user and status.
    
    Args:
        user_id: Filter by user ID.
        limit: Maximum number of jobs to return.
        status: Filter by job status.
        context: Execution context with user information.
    
    Returns:
        Dictionary with list of jobs:
        {
            "success": bool,
            "jobs": list[dict]
        }
    """
    print(f"User {context['user']['id']} is listing jobs...")
    
    try:
        # Default to current user's jobs if no user_id provided
        if not user_id and context and 'user' in context:
            user_id = context['user']['id']
            
        jobs = await job_queue.list_jobs(user_id, limit, status)
        
        return {
            "success": True,
            "jobs": jobs
        }
            
    except Exception as e:
        print(f"Error listing jobs: {e}")
        return {
            "success": False,
            "error": f"Failed to list jobs: {str(e)}"
        }


async def cancel_job(
    job_id: str,
    context: Any = None
) -> dict:
    """
    Cancels a job if it's still in the queue or running.
    
    Args:
        job_id: ID of the job to cancel.
        context: Execution context with user information.
    
    Returns:
        Dictionary with cancel result:
        {
            "success": bool,
            "job_id": str,
            "message": str,
            "status": str
        }
    """
    user_id = context['user']['id'] if context and 'user' in context else None
    print(f"User {user_id} is attempting to cancel job: {job_id}")
    
    try:
        # Call the job queue's cancel_job method with user validation
        result = await job_queue.cancel_job(job_id, user_id)
        return result
            
    except Exception as e:
        print(f"Error canceling job: {e}")
        return {
            "success": False,
            "job_id": job_id,
            "message": f"Failed to cancel job: {str(e)}",
            "status": None
        }


async def start_service(args, server):
    """Registers and runs the Hypha service."""
    service_id = args.service_id or f"conda-executor-{uuid.uuid4()}"
    svc_config = {
        "id": service_id,
        "name": service_id,
        "description": args.description,
        "type": "conda-python-executor",
        "config": {
            "visibility": "public",
            "require_context": True,
             # Let Hypha run the function in an executor thread pool
             # This is important as our function uses run_in_executor itself
            "run_in_executor": True,
        },
        # Expose service functions
        "execute": execute_in_conda_env,
        "submit_job": submit_job,
        "get_job_status": get_job_status,
        "get_job_result": get_job_result,
        "wait_for_result": wait_for_result,
        "list_jobs": list_jobs,
        "cancel_job": cancel_job,
    }
    try:
        service = await server.register_service(svc_config)
        service_url = f"{args.server_url}/{server.config.workspace}/services/{service.id.split('/')[-1]}"
        print(f"🔗 Service registered: {service.id}, URL: {service_url}")
        print("⏳ Service running. Press Ctrl+C to stop.")
        
        # Start the job queue worker
        await job_queue.start_worker()
        print("🚀 Job queue worker started")
        
        # Keep the service alive
        while True:
            await asyncio.sleep(3600)
    except Exception as e:
        print(f"🛑 Failed to register or run service: {e}")
        print(traceback.format_exc())
    finally:
        # Ensure we stop the worker when the service is shutting down
        await job_queue.stop_worker()
        print("🛑 Job queue worker stopped")


async def run():
    """Parses arguments, connects to Hypha, and starts the service."""
    parser = argparse.ArgumentParser(description="Hypha CondaEnvExecutor Service")
    parser.add_argument("--server-url", type=str, default="https://hypha.aicell.io",
                        help="Hypha server URL (default: https://hypha.aicell.io)")
    parser.add_argument("--token", type=str, default=os.environ.get("HYPHA_TOKEN"),
                        help="Hypha login token (can also be set via HYPHA_TOKEN env var)")
    parser.add_argument("--workspace", type=str, default=os.environ.get("HYPHA_WORKSPACE"),
                        help="Hypha workspace ID (can also be set via HYPHA_WORKSPACE env var)")
    parser.add_argument("--service-id", type=str, default=None,
                        help="Custom service ID (default: conda-executor-<uuid>)")
    parser.add_argument("--description", type=str,
                        default="Executes Python code in isolated Conda environments",
                        help="Service description")
    parser.add_argument("--job-queue-dir", type=str, default=None,
                        help="Directory to store job queue data (default: ~/.conda_env_jobs)")
    args = parser.parse_args()

    # Initialize job queue with custom storage dir if provided
    if args.job_queue_dir:
        global job_queue
        job_queue = JobQueue(args.job_queue_dir)

    token = args.token
    if not token:
        print("Hypha token not provided via --token or HYPHA_TOKEN env var.")
        print("Attempting interactive login...")
        try:
            # Assuming login can be interactive or handle server_url implicitly
            token = await login({"server_url": args.server_url})
        except Exception as e:
            print(f"🛑 Failed to login to Hypha: {e}")
            # print(traceback.format_exc()) # Avoid overly verbose output on common login issues
            return

    if not token:
        print("⚠️ Failed to obtain Hypha token. Exiting.")
        # return

    if not args.workspace:
        print("⚠️ Hypha workspace not provided via --workspace or HYPHA_WORKSPACE env var. Exiting.")
        # return

    try:
        print(f"🔌 Connecting to Hypha server: {args.server_url} (Workspace: {args.workspace})")
        server = await connect_to_server({
            "server_url": args.server_url,
            "token": token,
            "workspace": args.workspace,
            "client_id": "conda-executor-service" # Identify the client
        })
        print(f"✅ Connected to Hypha server!")

        await start_service(args, server)

    except ConnectionRefusedError:
        print(f"🛑 Connection refused. Is the Hypha server running at {args.server_url}?")
    except Exception as e:
        print(f"🛑 Failed to connect or start service: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n Shutdown requested by user. Exiting.")
    except Exception as e:
        print(f"\n An unexpected error occurred: {e}")
        print(traceback.format_exc())
