import asyncio
import argparse
import os
import traceback
import uuid
from typing import Union, Dict
import sys
from hypha_rpc import login, connect_to_server

# Ensure the executor module can be found, especially when running as a script
try:
    from conda_env_executor.executor import CondaEnvExecutor, ExecutionResult, TimingInfo
except ImportError:
    # Add the parent directory to sys.path to find the package
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PARENT_DIR = os.path.dirname(_CURRENT_DIR)
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)
    from conda_env_executor.executor import CondaEnvExecutor, ExecutionResult, TimingInfo


async def execute_in_conda_env(
    code: str,
    input_data: any = None,
    dependencies: list[Union[str, Dict]] = None,
    channels: list[str] = None,
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
            "require_context": False, # Does not require specific context object
             # Let Hypha run the function in an executor thread pool
             # This is important as our function uses run_in_executor itself
            "run_in_executor": True,
        },
        # Expose the main execution function
        "execute": execute_in_conda_env,
    }
    try:
        service = await server.register_service(svc_config)
        service_url = f"{args.server_url}/{server.config.workspace}/services/{service.id.split('/')[-1]}"
        print(f"🔗 Service registered: {service.id}, URL: {service_url}")
        print("⏳ Service running. Press Ctrl+C to stop.")
        # Keep the service alive
        while True:
            await asyncio.sleep(3600)
    except Exception as e:
        print(f"🛑 Failed to register or run service: {e}")
        print(traceback.format_exc())


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
    args = parser.parse_args()

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
