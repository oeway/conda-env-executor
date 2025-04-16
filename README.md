# Conda Environment Executor

A robust Python package for executing code in isolated conda environments with async support and job management.

## Features

- **Isolated Execution**: Run Python code in isolated conda environments
- **Async Support**: Execute code asynchronously with async/await syntax
- **Job Management**: Submit, monitor, cancel, and retrieve results from jobs
- **Timeout Handling**: Set execution timeouts for jobs
- **Shared Memory**: Efficient data transfer between processes using shared memory
- **Type Safety**: Full type safety with Pydantic models
- **Comprehensive Testing**: Extensive test coverage

## Installation

```bash
pip install conda-env-executor
```

## Quick Start

### Synchronous Execution

```python
from conda_env_executor import CondaEnvExecutor

# Create an executor with a conda environment
executor = CondaEnvExecutor(env_spec="environment.yml")

# Execute code
code = """
def execute(data=None):
    import numpy as np
    return np.array(data).mean()
"""

result = executor.execute(code, input_data=[1, 2, 3, 4, 5])
print(result.result)  # Output: 3.0
```

### Asynchronous Execution

```python
import asyncio
from conda_env_executor import AsyncCondaEnvExecutor, Job, ExecutionConfig

async def main():
    async with AsyncCondaEnvExecutor(env_spec="environment.yml") as executor:
        # Create a job
        job = Job(
            code="""
            def execute(data=None):
                import time
                time.sleep(2)  # Simulate long running task
                return data * 2
            """,
            input_data=21,
            config=ExecutionConfig(timeout=30)
        )
        
        # Submit the job
        job_id = await executor.submit_job(job)
        
        # Wait for result
        result = await executor.wait_for_result(job_id)
        print(result.result)  # Output: 42

asyncio.run(main())
```

### Using Shared Memory

```python
from conda_env_executor import CondaEnvExecutor, ExecutionConfig
import numpy as np

# Create large data
data = np.random.rand(1000000)

# Configure executor to use shared memory
executor = CondaEnvExecutor(
    env_spec="environment.yml",
    config=ExecutionConfig(use_shared_memory=True)
)

code = """
def execute(data):
    return data.mean()
"""

result = executor.execute(code, input_data=data)
print(result.result)
```

## Advanced Usage

### Job Management

```python
async def process_jobs():
    async with AsyncCondaEnvExecutor(env_spec="environment.yml") as executor:
        # Submit multiple jobs
        jobs = [
            Job(code="def execute(): return 1"),
            Job(code="def execute(): return 2"),
            Job(code="def execute(): return 3"),
        ]
        
        job_ids = []
        for job in jobs:
            job_id = await executor.submit_job(job)
            job_ids.append(job_id)
        
        # Monitor job status
        while job_ids:
            for job_id in job_ids[:]:
                status = await executor.get_job_status(job_id)
                if status.is_finished:
                    result = await executor.get_result(job_id)
                    print(f"Job {job_id}: {result.result}")
                    job_ids.remove(job_id)
            
            await asyncio.sleep(0.1)
```

### Error Handling

```python
from conda_env_executor import Job, ExecutionConfig

# Job with timeout
job = Job(
    code="""
    def execute():
        import time
        time.sleep(10)
        return 42
    """,
    config=ExecutionConfig(timeout=5)
)

try:
    result = executor.execute(job)
except TimeoutError:
    print("Job timed out")

# Job with retries
job = Job(
    code="""
    def execute():
        import random
        if random.random() < 0.5:
            raise ValueError("Random failure")
        return 42
    """,
    config=ExecutionConfig(max_retries=3, retry_delay=1.0)
)

result = executor.execute(job)
print(f"Took {result.metadata.attempts} attempts")
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/conda-env-executor.git
cd conda-env-executor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -e ".[dev,test]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=conda_env_executor

# Run specific test file
pytest tests/test_executor.py
```

### Code Quality

```bash
# Format code
black conda_env_executor tests

# Run linter
ruff check .

# Run type checker
mypy conda_env_executor
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Conda Environment Executor with Job Queue

This package provides a Hypha service for executing Python code in isolated Conda environments, with both synchronous and asynchronous execution options.

## Features

- **Isolated Execution**: Run Python code in a clean, isolated Conda environment with specified dependencies
- **Synchronous Execution**: Execute code and wait for results
- **Asynchronous Execution**: Submit jobs to a queue and retrieve results later
- **Job Management**: Submit, monitor, and retrieve results from jobs
- **Persistent Storage**: Job results are saved and can be retrieved even after service restart

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/conda-env-executor.git
cd conda-env-executor

# Install dependencies
pip install -e .
```

## Running the Service

Start the Hypha service:

```bash
python -m conda_env_executor.hypha_service --workspace YOUR_WORKSPACE
```

Options:
- `--server-url`: Hypha server URL (default: https://hypha.aicell.io)
- `--token`: Hypha login token (can also be set via HYPHA_TOKEN env var)
- `--workspace`: Hypha workspace ID (can also be set via HYPHA_WORKSPACE env var)
- `--service-id`: Custom service ID (default: conda-executor-<uuid>)
- `--job-queue-dir`: Directory to store job queue data (default: ~/.conda_env_jobs)

## Using the Service

### Synchronous Execution

To execute code synchronously and wait for results:

```python
import asyncio
from hypha_rpc import login, connect_to_server

async def main():
    # Connect to Hypha
    token = await login({"server_url": "https://hypha.aicell.io"})
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "token": token,
        "workspace": "YOUR_WORKSPACE"
    })
    
    service = await server.get_service("SERVICE_ID")
    
    # Execute code synchronously
    result = await service.execute(
        code="""
def execute(input_data):
    return {"message": "Hello from conda env!", "input": input_data}
        """,
        input_data={"name": "World"},
        dependencies=["python=3.9"]
    )
    
    print(result)

asyncio.run(main())
```

### Asynchronous Execution (Job Queue)

To submit a job and retrieve results later:

```python
import asyncio
from hypha_rpc import login, connect_to_server

async def main():
    # Connect to Hypha
    token = await login({"server_url": "https://hypha.aicell.io"})
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "token": token,
        "workspace": "YOUR_WORKSPACE"
    })
    
    service = await server.get_service("SERVICE_ID")
    
    # Submit a job
    submit_result = await service.submit_job(
        code="""
def execute(input_data):
    import time
    time.sleep(10)  # Simulate long-running task
    return {"message": "Task completed!", "input": input_data}
        """,
        input_data={"task": "long-running"},
        dependencies=["python=3.9"]
    )
    
    job_id = submit_result["job_id"]
    print(f"Job submitted with ID: {job_id}")
    
    # Check job status
    status = await service.get_job_status(job_id)
    print(f"Job status: {status}")
    
    # Wait for job to complete and get results
    result = await service.wait_for_result(job_id, timeout=60)
    print(f"Job result: {result}")
    
    # Or retrieve results later
    result = await service.get_job_result(job_id)
    print(f"Job result: {result}")
    
    # List all recent jobs
    jobs = await service.list_jobs()
    print(f"Recent jobs: {jobs}")
    
    # List only your jobs
    my_jobs = await service.list_jobs(user_id="me")
    print(f"My jobs: {my_jobs}")
    
    # Cancel a job
    cancel_result = await service.cancel_job(job_id)
    print(f"Cancel result: {cancel_result}")

asyncio.run(main())
```

### Job Management Features

The job queue system provides several key features for managing code execution jobs:

1. **User-Specific Jobs**: Each job is associated with the user who submitted it, allowing for:
   - Listing only your own jobs
   - User-based access control (only job owners can cancel their jobs)

2. **Job Status Tracking**: Monitor job status through the entire lifecycle:
   - Pending: Job is in queue waiting to be processed
   - Running: Job is currently being executed
   - Completed: Job has finished successfully
   - Failed: Job encountered an error or was canceled

3. **Job Cancellation**: Cancel jobs that are pending or running:
   - Pending jobs are immediately marked as canceled
   - Running jobs are marked for cancellation

4. **Persistent Storage**: All job information and results are saved to disk, allowing retrieval even after service restart

## Example Client

An example client script is provided to demonstrate job submission and monitoring:

```bash
# List all recent jobs
python examples/job_queue_client.py --workspace YOUR_WORKSPACE --service-id SERVICE_ID --list-jobs

# List only your jobs
python examples/job_queue_client.py --workspace YOUR_WORKSPACE --service-id SERVICE_ID --my-jobs

# List jobs filtered by status
python examples/job_queue_client.py --workspace YOUR_WORKSPACE --service-id SERVICE_ID --list-jobs --status running

# Submit a job
python examples/job_queue_client.py --workspace YOUR_WORKSPACE --service-id SERVICE_ID \
    --code-file examples/sample_job.py \
    --input-data examples/sample_input.json \
    --dependencies "python=3.9,numpy,pandas,matplotlib"

# Check job status
python examples/job_queue_client.py --workspace YOUR_WORKSPACE --service-id SERVICE_ID \
    --job-id JOB_ID

# Wait for job completion
python examples/job_queue_client.py --workspace YOUR_WORKSPACE --service-id SERVICE_ID \
    --job-id JOB_ID --wait

# Cancel a job
python examples/job_queue_client.py --workspace YOUR_WORKSPACE --service-id SERVICE_ID \
    --job-id JOB_ID --cancel
```

## API Reference

### Synchronous Execution

- **execute_in_conda_env**: Execute code in a conda environment and wait for results

### Asynchronous Execution (Job Queue)

- **submit_job**: Submit a job to the queue
- **get_job_status**: Check the status of a job
- **get_job_result**: Retrieve the result of a completed job
- **wait_for_result**: Wait for a job to complete and return its result
- **list_jobs**: List jobs, optionally filtered by user and status
- **cancel_job**: Cancel a job if it's still pending or running

# Conda Environment Executor

A Python package for executing code in isolated conda environments with efficient data passing between environments.

## Features

- Execute Python code in isolated conda environments
- Efficient data passing between environments using shared memory
- Support for conda-pack environments (portable tar.gz archives generated by [conda-pack](https://conda.github.io/conda-pack/))
- Environment caching for faster startup
- YAML-based environment specification
- Support for temporary environments
- Process safety with proper locking

## Installation

### Setting up the environment

First, create a Python 3.11 environment using conda or mamba:

```bash
# Using conda
conda create -n conda-env-executor python=3.11

# Or using mamba
mamba create -n conda-env-executor python=3.11

# Activate the environment
conda activate conda-env-executor
```

Then install the package:

```bash
pip install conda-env-executor
```

### Development Installation

For development, clone the repository and install with development dependencies:

```bash
git clone https://github.com/yourusername/conda-env-executor.git
cd conda-env-executor

# Install with development dependencies
pip install -e ".[dev,test]"
```

## Usage

### Getting Started

The simplest way to get started with Conda Environment Executor is to use temporary environments or YAML specifications:

#### Option 1: Using Temporary Environments

For quick testing or one-off executions, you can create temporary environments on-the-fly:

```python
from conda_env_executor import CondaEnvExecutor

# Create a temporary environment with specific packages
executor = CondaEnvExecutor.create_temp_env(['python=3.11', 'numpy', 'pandas'])

# Define code to run in the environment
code = """
import numpy as np
import pandas as pd

def process_data(data):
    return pd.DataFrame(data).describe().to_dict()

result = process_data(input_data)
"""

# Prepare input data
input_data = {"values": [1, 2, 3, 4, 5]}

# Execute code in the temporary environment
with executor:
    result = executor.execute(code, input_data)

print(result)
```

The temporary environment will be automatically created and cleaned up when no longer needed.

#### Option 2: Using Environment Specifications

For more reproducible workflows, define your environment in a YAML file:

```yaml
# environment.yaml
name: myenv
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - pandas
```

Then use it with the executor:

```python
from conda_env_executor import CondaEnvExecutor

executor = CondaEnvExecutor.from_yaml('environment.yaml')
with executor:
    result = executor.execute(code, input_data)
```

This approach creates and manages the environment for you without requiring manual packaging steps.

### Advanced Usage: Using Conda-Pack Environments

For production deployments or environments that need to be shared across machines, you can use [conda-pack](https://conda.github.io/conda-pack/) to create portable environment archives.

#### Packaging Environments with Conda-Pack

First, create a conda environment and package it:

```bash
# Create a conda environment
conda create -n myenv python=3.11 numpy pandas

# Activate the environment and install any additional packages
conda activate myenv
conda install scikit-learn matplotlib

# Package the environment into a portable archive using conda-pack
conda pack -n myenv -o myenv.tar.gz
```

Then use the packaged environment with the executor:

```python
from conda_env_executor import CondaEnvExecutor

# Create an executor using the packed environment
executor = CondaEnvExecutor("myenv.tar.gz")

# Define some code to run in the environment
code = """
import numpy as np
import pandas as pd

def process_data(data):
    df = pd.DataFrame(data)
    return df.describe().to_dict()

result = process_data(input_data)
"""

# Prepare input data
input_data = {"values": [1, 2, 3, 4, 5]}

# Execute the code in the isolated environment
with executor:
    result = executor.execute(code, input_data)

print(result)
```

Benefits of using conda-pack:
- Create once, deploy many times
- No internet connection needed for deployment
- Exact package versions preserved
- Faster startup times for complex environments
- Ideal for production or shared environments

## Conda-Pack Tutorial

[Conda-pack](https://conda.github.io/conda-pack/) is a tool for creating relocatable conda environments. This is useful for deploying code in an isolated environment, copying environments to a different location or machine, or for archiving environments.

### 1. Install conda-pack

First, you need to install conda-pack:

```bash
# Install in your base environment
pip install conda-pack

# Or install in a specific environment
conda install -c conda-forge conda-pack
```

### 2. Create and Set Up Your Environment

Create a conda environment with the packages you need:

```bash
# Create a new environment
conda create -n myenv python=3.11 numpy pandas scikit-learn matplotlib

# Activate the environment
conda activate myenv

# Install any additional packages
pip install some-package
```

### 3. Package Your Environment

Once your environment is set up with all required packages, use conda-pack to create a portable archive:

```bash
# Basic packaging (from outside the environment)
conda pack -n myenv -o myenv.tar.gz

# Or if you're inside the environment
conda pack -o myenv.tar.gz

# For more verbose output
conda pack -n myenv -o myenv.tar.gz --verbose
```

### 4. Using the Packed Environment

There are multiple ways to use the packed environment:

#### Option A: Unpacking for direct use (on the same architecture)

```bash
# Create a directory for the environment
mkdir -p /path/to/extract/myenv

# Extract the environment
tar -xzf myenv.tar.gz -C /path/to/extract/myenv

# Activate the environment
source /path/to/extract/myenv/bin/activate

# When you're done
source /path/to/extract/myenv/bin/deactivate
```

#### Option B: Using with conda-env-executor (this package)

Use the packed environment with conda-env-executor as shown in the Basic Usage section above.

### 5. Troubleshooting Conda Packs

If you encounter issues with your packed environment:

- Make sure all dependencies are properly installed in the original environment
- Try packaging with `--ignore-editable` if you have editable packages
- Use `--ignore-missing-files` if there are path conflicts
- For compatibility across different systems, pack from a similar OS/architecture as the target system

### 6. Cleaning Up

After packaging, you can clean up temporary files created during packaging:

```bash
# Clean up the prefixes directory in the original environment
conda pack -n myenv --clean
```

## Requirements

- Python >=3.10
- pyyaml >=6.0
- psutil >=5.9.0
- conda-pack >=0.7.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For development, you'll need additional dependencies that can be installed with:
```bash
pip install -e ".[dev,test]"
```

This will install:
- Testing: pytest, pytest-cov, numpy
- Development: black, isort, mypy

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project incorporates ideas and code from:
- [conda-execute](https://github.com/conda-tools/conda-execute) (BSD 3-Clause License)
- [conda-pack](https://github.com/conda/conda-pack) (BSD 3-Clause License)

# SandboxAI Evaluation

This repository contains a test script to evaluate the capabilities of SandboxAI, particularly focusing on:
- Basic sandbox functionality
- Package installation
- Image analysis using scikit-image
- Machine learning using scikit-learn
- Plotting capabilities with matplotlib

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the test script:
```bash
python sandbox_test.py
```

## Test Cases

The script includes three main test cases:

1. **Basic Sandbox Test**: Tests basic Python execution and package installation within the sandbox.

2. **Image Analysis Test**: Demonstrates image processing capabilities using scikit-image:
   - Loads a sample image
   - Converts to grayscale
   - Applies Gaussian blur
   - Calculates image statistics
   - Generates comparison plots (saved as 'image_analysis.png')

3. **Machine Learning Test**: Shows machine learning capabilities using scikit-learn:
   - Creates a synthetic classification dataset
   - Trains a Random Forest classifier
   - Evaluates model performance
   - Plots feature importance (saved as 'feature_importance.png')

## Output

The script will generate:
- Console output with test results
- Two image files:
  - `image_analysis.png`: Showing image processing results
  - `feature_importance.png`: Showing feature importance from the ML model

# Docker-based Python Sandbox

This repository contains a test script that demonstrates how to create isolated Python environments using Docker containers. The implementation focuses on:
- Running Python code in isolated containers
- Installing and using various Python packages
- Performing image analysis and machine learning tasks
- Generating and saving plots and results

## Requirements

1. Docker must be installed and running on your system
2. Python 3.6+ with pip

## Setup

1. Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Docker daemon is running

3. Run the test script:
```bash
python docker_sandbox_test.py
```

## Test Cases

The script includes three main test cases:

1. **Basic Sandbox Test**: Tests basic Python code execution in an isolated container.

2. **Image Analysis Test**: Demonstrates image processing capabilities:
   - Loads a sample image from scikit-image
   - Converts to grayscale
   - Applies Gaussian blur
   - Calculates image statistics
   - Generates comparison plots (saved as 'image_analysis.png')

3. **Machine Learning Test**: Shows machine learning capabilities:
   - Creates a synthetic classification dataset
   - Trains a Random Forest classifier
   - Evaluates model performance
   - Plots feature importance (saved as 'feature_importance.png')

## How it Works

The `DockerSandbox` class provides the following functionality:
- Creates temporary directories for code execution
- Builds custom Docker images with required dependencies
- Runs code in isolated containers
- Captures output and saves generated files
- Automatically cleans up containers and temporary files

## Output

The script generates:
- Console output with test results
- Two image files in the temporary directory:
  - `image_analysis.png`: Showing image processing results
  - `feature_importance.png`: Showing feature importance from the ML model

## Security Notes

This implementation provides isolation through Docker containers, which offers several benefits:
- Isolated filesystem
- Controlled resource usage
- Clean environment for each execution
- Safe package installation and management