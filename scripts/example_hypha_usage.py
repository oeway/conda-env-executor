import asyncio
import inspect
import textwrap
import os
import argparse
import uuid
import sys
from hypha_rpc import connect_to_server, login


def remote_conda_function(
    service_id: str,
    env_spec_type: str = 'packages',
    env_spec_value: any = None,
    channels: list[str] = None,
):
    """
    Decorator to execute a Python function remotely within a Conda environment
    using the conda-python-executor Hypha service.

    Args:
        service_id: The full ID of the target conda-python-executor service
                    (e.g., "workspace/conda-executor-uuid").
        env_spec_type: Type of environment specification ('yaml_file',
                       'pack_file', 'yaml_content', 'packages').
        env_spec_value: The specification value corresponding to the type.
                       Defaults to ['python'] if type is 'packages' and value is None.
        channels: List of Conda channels (used for 'packages' type).
                  Defaults to ['conda-forge'].
    """
    if env_spec_type == 'packages' and env_spec_value is None:
        env_spec_value = ["python"] # Default minimal environment

    if env_spec_type == 'packages' and channels is None:
        channels = ["conda-forge"] # Default channels

    def decorator(func):
        # Get the source code of the decorated function
        try:
                source_lines, _ = inspect.getsourcelines(func)
                def_index = next(
                    i for i, line in enumerate(source_lines)
                    if line.strip().startswith("def ") or line.strip().startswith("async def ")
                )
                cleaned_lines = source_lines[def_index:]
                source_code = textwrap.dedent("".join(cleaned_lines))
        except OSError:
            raise TypeError(f"Cannot get source code for function {func.__name__}. "
                            "Is it defined dynamically or in an interactive session?")

        # Ensure the decorated function is async
        if not asyncio.iscoroutinefunction(func):
             raise TypeError(f"Decorated function {func.__name__} must be an async function (defined with 'async def').")


        async def wrapper(*args, **kwargs):
            """The wrapper that calls the Hypha service."""
            # We need to pass args/kwargs as input_data
            input_data = {"args": args, "kwargs": kwargs}

            # Construct the code to be executed remotely.
            # It defines the necessary 'execute' function which then calls
            # the user's original async function.
            # Note: This assumes the user function's dependencies are either
            # standard library or installed via the conda env spec.
            # It also assumes the user function is defined at the top level
            # or its definition doesn't rely on complex closures captured
            # from the surrounding scope that aren't serializable/reproducible
            # from source alone.
            remote_code = f"""import asyncio
{source_code}
def execute(input_data):
    # Unpack args and kwargs
    args = input_data.get('args', [])
    kwargs = input_data.get('kwargs', {{}})

    # Call the user's original async function
    import asyncio
    result = asyncio.run({func.__name__}(*args, **kwargs))
    return result
"""
            # Get Hypha server connection (assuming server is available globally or passed in)
            # In a real application, handle server connection more robustly.
            if 'server' not in globals():
                 raise RuntimeError("Hypha server connection not available in global scope.")

            try:
                print(f"Attempting to get service with ID: '{service_id}'")
                service = await server.get_service(service_id)
                print(f"✅ Successfully found service.")
            except Exception as e:
                print(f"❌ Failed to get service '{service_id}': {e}")
                print(f"Server object: {server}")
                print(f"Available services: {await server.list_services()}")
                return None

            print(f"🚀 Calling remote function '{func.__name__}' via service '{service_id}'...")
            print(f"   Environment Type: {env_spec_type}")
            print(f"   Environment Spec: {str(env_spec_value)[:100]}...")
            if channels:
                 print(f"   Channels: {channels}")

            # Call the Hypha service's execute method
            result = await service.execute(
                code=remote_code,
                input_data=input_data,
                env_spec_type=env_spec_type,
                env_spec_value=env_spec_value,
                channels=channels,
            )

            # Handle response
            if result.get("stdout"):
                print("--- Remote Stdout ---")
                print(result["stdout"].strip())
                print("---------------------")
            if result.get("stderr"):
                print("--- Remote Stderr ---", file=sys.stderr)
                print(result["stderr"].strip(), file=sys.stderr)
                print("---------------------", file=sys.stderr)

            if result.get("success"):
                print(f"✅ Remote function '{func.__name__}' executed successfully.")
                if result.get("timing"):
                    print(f"   Timing: {result['timing']}")
                return result.get("result")
            else:
                print(f"❌ ERROR executing remote function '{func.__name__}':")
                print(f"   Error: {result.get('error')}")
                if result.get('traceback'):
                    print("--- Remote Traceback ---")
                    print(result['traceback'].strip())
                    print("------------------------")
                return None # Or raise an exception

        return wrapper
    return decorator



async def main():
    global server # Allow modifying the global server variable

    parser = argparse.ArgumentParser(description="Conda Executor Hypha Service Example")
    parser.add_argument("--server-url", type=str, default="https://hypha.aicell.io",
                        help="Hypha server URL")
    parser.add_argument("--workspace", type=str, default=os.environ.get("HYPHA_WORKSPACE"),
                        help="Hypha workspace ID")
    parser.add_argument("--token", type=str, default=os.environ.get("HYPHA_TOKEN"),
                        help="Hypha login token")
    parser.add_argument("--service-id", type=str, required=True,
                        help="Full ID of the running conda-executor service (e.g., 'your-workspace/conda-executor-uuid')")

    args = parser.parse_args()

    print(f"Service ID: {args.service_id}")
    print(f"Workspace: {args.workspace}")
    print(f"Server URL: {args.server_url}")
    
    token = args.token
    # if not token:
    #     print("Token not provided, attempting login...")
    #     try:
    #         token = await login({"server_url": args.server_url})
    #     except Exception as e:
    #         print(f"🛑 Login failed: {e}")
    #         return
    #     if not token:
    #         print("🛑 Could not obtain token.")
    #         return

    print(f"🔌 Connecting to Hypha: {args.server_url} (Workspace: {args.workspace})")
    try:
        server = await connect_to_server({
            "server_url": args.server_url,
            "token": token,
            "workspace": args.workspace,
            "client_id": f"conda-executor-client-{uuid.uuid4()}"
        })
        print("✅ Connected to Hypha.")
    except Exception as e:
        print(f"🛑 Connection failed: {e}")
        return

    # --- Apply the decorator ---
    # Decorate the function, specifying the service and required environment
    @remote_conda_function(
        service_id=args.service_id,
        env_spec_type='packages',
        env_spec_value=['python=3.9', 'numpy'] # Specify dependencies
    )
    async def remote_process_data(data: list, multiplier: int = 1):
        """Example function that uses numpy (needs to be in the env spec)."""
        # NOTE: Imports needed by this function must be included in the
        # conda environment specification.
        import numpy as np
        print(f"Received data: {data}")
        print(f"Multiplier: {multiplier}")
        arr = np.array(data)
        result = np.mean(arr) * multiplier
        print(f"Calculated mean * multiplier: {result}")
        return {"mean_multiplied": result, "original_shape": arr.shape}
    

    # --- Call the decorated function ---
    print("\n--- Calling decorated function 'remote_process_data' ---")
    result = await remote_process_data([1, 2, 3, 4, 5], multiplier=10)

    if result is not None:
        print("\n--- Result received by client ---")
        print(result)
        print("-------------------------------")

    # --- Example with different args ---
    print("\n--- Calling again with different args ---")
    result2 = await remote_process_data([10, 20, 30]) # multiplier defaults to 1

    if result2 is not None:
        print("\n--- Result 2 received by client ---")
        print(result2)
        print("---------------------------------")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient interrupted.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc() 