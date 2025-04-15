#!/usr/bin/env python
"""
Example client for the conda environment executor job queue functionality.

This script demonstrates how to submit jobs, check status, and retrieve results
using the Hypha client interface.
"""

import asyncio
import argparse
import json
import time
from pprint import pprint
from hypha_rpc import login, connect_to_server


async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Conda Env Executor Job Queue Client")
    parser.add_argument("--server-url", type=str, default="https://hypha.aicell.io",
                        help="Hypha server URL")
    parser.add_argument("--workspace", type=str, required=True,
                        help="Hypha workspace ID")
    parser.add_argument("--service-id", type=str, required=True,
                        help="ID of the registered conda executor service")
    parser.add_argument("--code-file", type=str,
                        help="Path to Python file containing code to execute")
    parser.add_argument("--input-data", type=str,
                        help="JSON string or path to JSON file with input data")
    parser.add_argument("--dependencies", type=str, default="python",
                        help="Comma-separated list of conda/pip packages")
    parser.add_argument("--channels", type=str, default="conda-forge",
                        help="Comma-separated list of conda channels")
    parser.add_argument("--list-jobs", action="store_true",
                        help="List recent jobs")
    parser.add_argument("--my-jobs", action="store_true",
                        help="List only the current user's jobs")
    parser.add_argument("--status", type=str, choices=["pending", "running", "completed", "failed"],
                        help="Filter jobs by status")
    parser.add_argument("--job-id", type=str,
                        help="Job ID to check status or retrieve results")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for job completion")
    parser.add_argument("--cancel", action="store_true",
                        help="Cancel the specified job")
    parser.add_argument("--timeout", type=float, default=300,
                        help="Timeout in seconds when waiting for job completion")
    parser.add_argument("--limit", type=int, default=10,
                        help="Maximum number of jobs to list")
    args = parser.parse_args()

    # Handle login
    token = await login({"server_url": args.server_url})
    if not token:
        print("Failed to login to Hypha server.")
        return

    # Connect to Hypha server
    server = await connect_to_server({
        "server_url": args.server_url,
        "token": token,
        "workspace": args.workspace
    })
    
    # Get service object
    service = await server.get_service(args.service_id)
    if not service:
        print(f"Service {args.service_id} not found in workspace {args.workspace}")
        return
    
    print(f"Connected to service: {service.id}")

    # Process commands
    if args.list_jobs or args.my_jobs:
        # If --my-jobs is specified, only show current user's jobs
        user_id = "me" if args.my_jobs else None
        await list_jobs(service, user_id, args.limit, args.status)
        
    elif args.job_id:
        if args.cancel:
            await cancel_job(service, args.job_id)
        elif args.wait:
            await wait_for_job(service, args.job_id, args.timeout)
        else:
            await check_job(service, args.job_id)
            
    elif args.code_file:
        await submit_new_job(service, args)
        
    else:
        print("No action specified. Use:")
        print("  --code-file to submit a job")
        print("  --job-id to check job status")
        print("  --job-id --cancel to cancel a job")
        print("  --list-jobs to list all jobs")
        print("  --my-jobs to list your jobs only")


async def list_jobs(service, user_id=None, limit=10, status=None):
    """List jobs, optionally filtered by user and status."""
    print("\n=== Listing Jobs ===")
    
    user_label = "your" if user_id == "me" else "all"
    status_label = f" with status '{status}'" if status else ""
    print(f"Retrieving {user_label} jobs{status_label}...")
    
    result = await service.list_jobs(user_id=user_id, limit=limit, status=status)
    
    if not result.get("success"):
        print(f"Failed to list jobs: {result.get('error')}")
        return
    
    jobs = result.get("jobs", [])
    if not jobs:
        print("No jobs found.")
        return
    
    print(f"Found {len(jobs)} jobs:")
    for job in jobs:
        print(f"- Job ID: {job['job_id']}")
        print(f"  Status: {job['status']}")
        print(f"  User: {job.get('user_id', 'Unknown')}")
        print(f"  Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job['created_at']))}")
        if job.get('completed_at'):
            print(f"  Completed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job['completed_at']))}")
        print()


async def check_job(service, job_id):
    """Check job status and possibly retrieve results."""
    print(f"\n=== Checking Job {job_id} ===")
    
    # Get job status
    status_result = await service.get_job_status(job_id)
    if not status_result.get("success"):
        print(f"Failed to check job status: {status_result.get('error')}")
        return
    
    status = status_result["status"]
    print(f"Job Status: {status['status']}")
    print(f"User: {status.get('user_id', 'Unknown')}")
    print(f"Created At: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(status['created_at']))}")
    
    if status["status"] in ("completed", "failed"):
        print(f"Completed At: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(status['completed_at']))}")
        
        # Get job result
        result = await service.get_job_result(job_id)
        if not result.get("success"):
            print(f"Failed to retrieve job result: {result.get('error')}")
            return
        
        print("\n=== Job Result ===")
        job_result = result["result"]
        
        print(f"Success: {job_result['success']}")
        
        if job_result.get("error"):
            print(f"Error: {job_result['error']}")
        
        if job_result.get("stdout"):
            print("\n--- Standard Output ---")
            print(job_result["stdout"])
        
        if job_result.get("stderr"):
            print("\n--- Standard Error ---")
            print(job_result["stderr"])
        
        if job_result.get("result"):
            print("\n--- Result ---")
            pprint(job_result["result"])
        
        if job_result.get("timing"):
            print("\n--- Timing Information ---")
            timing = job_result["timing"]
            print(f"Environment Setup Time: {timing.get('env_setup_time', 'N/A')} seconds")
            print(f"Execution Time: {timing.get('execution_time', 'N/A')} seconds")
            print(f"Total Time: {timing.get('total_time', 'N/A')} seconds")


async def wait_for_job(service, job_id, timeout):
    """Wait for job completion and show results."""
    print(f"\n=== Waiting for Job {job_id} (timeout: {timeout}s) ===")
    
    result = await service.wait_for_result(job_id, timeout)
    
    if not result.get("success"):
        print(f"Failed to wait for job result: {result.get('error')}")
        if result.get("status"):
            print(f"Current job status: {result['status']['status']}")
        return
    
    print("Job completed!")
    job_result = result["result"]
    
    print(f"Success: {job_result['success']}")
    
    if job_result.get("error"):
        print(f"Error: {job_result['error']}")
    
    if job_result.get("stdout"):
        print("\n--- Standard Output ---")
        print(job_result["stdout"])
    
    if job_result.get("stderr"):
        print("\n--- Standard Error ---")
        print(job_result["stderr"])
    
    if job_result.get("result"):
        print("\n--- Result ---")
        pprint(job_result["result"])
    
    if job_result.get("timing"):
        print("\n--- Timing Information ---")
        timing = job_result["timing"]
        print(f"Environment Setup Time: {timing.get('env_setup_time', 'N/A')} seconds")
        print(f"Execution Time: {timing.get('execution_time', 'N/A')} seconds")
        print(f"Total Time: {timing.get('total_time', 'N/A')} seconds")


async def cancel_job(service, job_id):
    """Cancel a job."""
    print(f"\n=== Canceling Job {job_id} ===")
    
    result = await service.cancel_job(job_id)
    
    if result.get("success"):
        print(f"Job canceled successfully: {result.get('message')}")
        print(f"Current status: {result.get('status')}")
    else:
        print(f"Failed to cancel job: {result.get('message')}")
        print(f"Current status: {result.get('status')}")


async def submit_new_job(service, args):
    """Submit a new job to the queue."""
    print("\n=== Submitting New Job ===")
    
    # Read code from file
    with open(args.code_file, 'r') as f:
        code = f.read()
    
    # Parse input data
    input_data = None
    if args.input_data:
        if args.input_data.startswith('{') or args.input_data.startswith('['):
            # Try to parse as JSON string
            try:
                input_data = json.loads(args.input_data)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in input data")
                return
        else:
            # Try to read from file
            try:
                with open(args.input_data, 'r') as f:
                    input_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error reading input data file: {e}")
                return
    
    # Parse dependencies and channels
    dependencies = [dep.strip() for dep in args.dependencies.split(',')]
    channels = [channel.strip() for channel in args.channels.split(',')]
    
    # Submit job
    result = await service.submit_job(
        code=code,
        input_data=input_data,
        dependencies=dependencies,
        channels=channels
    )
    
    if not result.get("success"):
        print(f"Failed to submit job: {result.get('error')}")
        return
    
    job_id = result["job_id"]
    status = result["status"]
    
    print(f"Job submitted successfully!")
    print(f"Job ID: {job_id}")
    print(f"Status: {status['status']}")
    print(f"Created At: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(status['created_at']))}")
    
    print("\nAvailable commands:")
    print(f"  Check status: python {__file__} --job-id {job_id} --server-url {args.server_url} --workspace {args.workspace} --service-id {args.service_id}")
    print(f"  Wait for completion: python {__file__} --job-id {job_id} --wait --server-url {args.server_url} --workspace {args.workspace} --service-id {args.service_id}")
    print(f"  Cancel job: python {__file__} --job-id {job_id} --cancel --server-url {args.server_url} --workspace {args.workspace} --service-id {args.service_id}")


if __name__ == "__main__":
    asyncio.run(main()) 