# Conda Environment Executor

A robust Python package for executing code in isolated conda environments with efficient data passing and multiple environment creation options.

## Features

- **Isolated Execution**: Run Python code in isolated conda environments
- **Multiple Environment Sources**: Support for conda-pack files, YAML specifications, dictionaries, and temporary environments
- **Efficient Data Passing**: Optimized data transfer between environments using JSON serialization with NumPy support
- **Environment Caching**: Automatic caching of environments for faster subsequent executions
- **Async Support**: Built-in support for async/await patterns through Hypha service integration
- **Job Management**: Submit, monitor, cancel, and retrieve results from jobs via job queue system
- **Type Safety**: Full type safety with proper error handling and result objects
- **Comprehensive Testing**: Extensive test coverage

## Installation

```bash
pip install conda-env-executor
```

## Quick Start

### 1. Temporary Environments (Simplest)

The easiest way to get started is creating temporary environments on-the-fly:

```python
from conda_env_executor import CondaEnvExecutor

# Create a temporary environment with specific packages
executor = CondaEnvExecutor.create_temp_env(
    packages=['python=3.11', 'numpy', 'pandas'],
    channels=['conda-forge']
)

# Define code to run - must include an 'execute' function
code = """
import numpy as np
import pandas as pd

def execute(data):
    df = pd.DataFrame(data)
    return {
        'mean': df.values.mean(),
        'shape': df.shape,
        'description': df.describe().to_dict()
    }
"""

# Execute with input data
input_data = {"values": [1, 2, 3, 4, 5], "names": ["a", "b", "c", "d", "e"]}

with executor:
    result = executor.execute(code, input_data)
    if result.success:
        print(result.result)
    else:
        print(f"Error: {result.error}")
```

### 2. YAML Environment Specifications

For reproducible environments, use YAML specifications:

```yaml
# environment.yml
name: data-analysis
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy>=1.20
  - pandas>=1.3
  - scikit-learn
  - matplotlib
  - pip
  - pip:
    - some-pip-package
```

```python
from conda_env_executor import CondaEnvExecutor

# Method 1: Direct path
executor = CondaEnvExecutor("environment.yml")

# Method 2: Using class method
executor = CondaEnvExecutor.from_yaml("environment.yml")

# Execute code
with executor:
    result = executor.execute(code, input_data)
```

### 3. Dictionary Specifications

Define environments programmatically:

```python
from conda_env_executor import CondaEnvExecutor

# Define environment as dictionary
env_spec = {
    "name": "ml-env",
    "channels": ["conda-forge", "pytorch"],
    "dependencies": [
        "python=3.11",
        "numpy",
        "pandas",
        "pytorch",
        "scikit-learn",
        {"pip": ["transformers", "datasets"]}
    ]
}

executor = CondaEnvExecutor(env_spec)

with executor:
    result = executor.execute(code, input_data)
```

## Advanced Usage Patterns

### 4. Conda-Pack Files (Production Deployments)

For production environments or when you need to share exact environments across machines:

```python
from conda_env_executor import CondaEnvExecutor

# Use a pre-built conda-pack file
executor = CondaEnvExecutor("myenv.tar.gz")

with executor:
    result = executor.execute(code, input_data)
```

To create conda-pack files:
```bash
# Create environment
conda create -n myenv python=3.11 numpy pandas scikit-learn
conda activate myenv

# Package the environment
conda install conda-pack
conda pack -n myenv -o myenv.tar.gz
```

### 5. Data Handling Patterns

The executor handles various data types automatically:

```python
import numpy as np
from conda_env_executor import CondaEnvExecutor

executor = CondaEnvExecutor.create_temp_env(['python=3.11', 'numpy'])

# NumPy arrays are automatically serialized/deserialized
data = np.random.rand(1000, 10)

code = """
import numpy as np

def execute(data):
    # data is automatically converted back to numpy array
    return {
        'mean': float(data.mean()),
        'shape': list(data.shape),
        'std': float(data.std())
    }
"""

with executor:
    result = executor.execute(code, data)
    print(result.result)  # {'mean': 0.5, 'shape': [1000, 10], 'std': 0.29}
```

### 6. Complex Dependencies

Handle pip packages and mixed dependencies:

```python
# Complex environment with conda and pip packages
executor = CondaEnvExecutor.create_temp_env(
    packages=[
        'python=3.11',
        'numpy',
        'pandas',
        'matplotlib',
        {'pip': [
            'transformers>=4.20.0',
            'datasets',
            'torch-audio'
        ]}
    ],
    channels=['conda-forge', 'pytorch']
)

code = """
import pandas as pd
from transformers import pipeline

def execute(texts):
    # Use Hugging Face transformers
    classifier = pipeline("sentiment-analysis")
    results = classifier(texts)
    
    # Convert to pandas for analysis
    df = pd.DataFrame(results)
    return df.to_dict('records')
"""

with executor:
    result = executor.execute(code, ["I love this!", "This is terrible"])
```

### 7. Environment Reuse and Caching

Environments are automatically cached for performance:

```python
# First execution - environment is created
executor1 = CondaEnvExecutor.create_temp_env(['python=3.11', 'numpy'])
with executor1:
    result1 = executor1.execute(code, data)  # Slow first time

# Second execution - environment is reused from cache
executor2 = CondaEnvExecutor.create_temp_env(['python=3.11', 'numpy'])
with executor2:
    result2 = executor2.execute(code, data)  # Fast subsequent times
```

### 8. Error Handling and Debugging

Comprehensive error handling with timing information:

```python
from conda_env_executor import CondaEnvExecutor

executor = CondaEnvExecutor.create_temp_env(['python=3.11'])

# Code with an error
bad_code = """
def execute(data):
    return undefined_variable  # This will cause an error
"""

with executor:
    result = executor.execute(bad_code, {"test": "data"})
    
    if result.success:
        print(f"Result: {result.result}")
    else:
        print(f"Execution failed: {result.error}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        
    # Timing information
    if result.timing:
        print(f"Environment setup: {result.timing.env_setup_time:.2f}s")
        print(f"Code execution: {result.timing.execution_time:.2f}s")
        print(f"Total time: {result.timing.total_time:.2f}s")
```

### 9. Multiple Executions with Same Environment

Reuse the same environment for multiple code executions:

```python
executor = CondaEnvExecutor.create_temp_env(['python=3.11', 'numpy', 'pandas'])

# Execute multiple different pieces of code
codes = [
    "def execute(data): import numpy as np; return np.mean(data)",
    "def execute(data): import pandas as pd; return pd.Series(data).describe().to_dict()",
    "def execute(data): return {'sum': sum(data), 'len': len(data)}"
]

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

with executor:  # Environment is set up once
    for i, code in enumerate(codes):
        result = executor.execute(code, data)
        print(f"Result {i+1}: {result.result}")
```

## Complete API Reference

### Core Classes

#### CondaEnvExecutor

Main class for executing code in conda environments.

**Constructor Options:**
```python
# From conda-pack file
CondaEnvExecutor("path/to/env.tar.gz")

# From YAML file
CondaEnvExecutor("environment.yml")

# From dictionary
CondaEnvExecutor({
    "name": "myenv",
    "channels": ["conda-forge"],
    "dependencies": ["python=3.11", "numpy"]
})

# From EnvSpec object
CondaEnvExecutor(EnvSpec(...))
```

**Class Methods:**
```python
# Create temporary environment
CondaEnvExecutor.create_temp_env(
    packages=["python=3.11", "numpy"],
    channels=["conda-forge"]
)

# Create from YAML file
CondaEnvExecutor.from_yaml("environment.yml")
```

**Instance Methods:**
```python
# Execute code
result = executor.execute(code, input_data=None)

# Manual cleanup
executor.cleanup()

# Context manager (automatic cleanup)
with executor:
    result = executor.execute(code, input_data)
```

#### ExecutionResult

Container for execution results:

```python
@dataclass
class ExecutionResult:
    success: bool                    # Whether execution succeeded
    result: Optional[Any] = None     # The returned result
    error: Optional[str] = None      # Error message if failed
    stdout: Optional[str] = None     # Standard output
    stderr: Optional[str] = None     # Standard error
    timing: Optional[TimingInfo] = None  # Timing information
```

#### TimingInfo

Timing information for execution:

```python
@dataclass
class TimingInfo:
    env_setup_time: float    # Time to set up environment
    execution_time: float    # Time to execute code
    total_time: float       # Total execution time
```

### Dependency Specification Formats

#### List Format
```python
packages = [
    "python=3.11",           # Specific version
    "numpy>=1.20",           # Version constraint
    "pandas",                # Latest version
    {"pip": ["requests"]},   # Pip packages
    {"pip": [               # Multiple pip packages
        "transformers>=4.20.0",
        "datasets"
    ]}
]
```

#### Dictionary Format (environment.yml style)
```python
env_spec = {
    "name": "myenv",
    "channels": ["conda-forge", "pytorch"],
    "dependencies": [
        "python=3.11",
        "numpy",
        {"pip": ["requests", "beautifulsoup4"]}
    ]
}
```

### Code Requirements

Your code must define an `execute` function:

```python
def execute(input_data):
    """
    This function will be called by the executor.
    
    Args:
        input_data: The data passed to executor.execute()
                   Can be None if no input_data provided
    
    Returns:
        Any JSON-serializable object
    """
    # Your code here
    return result
```

## Job Queue System (Hypha Service)

For async execution and job management, use the Hypha service:

### Starting the Service

```bash
python -m conda_env_executor.hypha_service \
    --workspace YOUR_WORKSPACE \
    --server-url https://hypha.aicell.io \
    --token YOUR_TOKEN
```

### Using the Service

```python
import asyncio
from hypha_rpc import connect_to_server

async def main():
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "token": "your_token",
        "workspace": "your_workspace"
    })
    
    service = await server.get_service("conda-executor-service-id")
    
    # Submit a job
    result = await service.submit_job(
        code="def execute(data): return data * 2",
        input_data=21,
        dependencies=["python=3.11", "numpy"]
    )
    
    job_id = result["job_id"]
    
    # Wait for completion
    final_result = await service.wait_for_result(job_id)
    print(final_result)  # 42

asyncio.run(main())
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
pytest tests/test_executor.py -v
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

## Job Queue Examples

Example client usage for job management:

```bash
# Submit a job
python examples/job_queue_client.py --workspace YOUR_WORKSPACE --service-id SERVICE_ID \
    --code-file examples/sample_job.py \
    --input-data examples/sample_input.json \
    --dependencies "python=3.11,numpy,pandas,matplotlib"

# List jobs
python examples/job_queue_client.py --workspace YOUR_WORKSPACE --service-id SERVICE_ID --list-jobs

# Check job status and wait for completion
python examples/job_queue_client.py --workspace YOUR_WORKSPACE --service-id SERVICE_ID \
    --job-id JOB_ID --wait
```

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

Use the packed environment with conda-env-executor:

```python
from conda_env_executor import CondaEnvExecutor

# Create an executor using the packed environment
executor = CondaEnvExecutor("myenv.tar.gz")

with executor:
    result = executor.execute(code, input_data)
```

### 5. Troubleshooting Conda Packs

If you encounter issues with your packed environment:

- Make sure all dependencies are properly installed in the original environment
- Try packaging with `--ignore-editable` if you have editable packages
- Use `--ignore-missing-files` if there are path conflicts
- For compatibility across different systems, pack from a similar OS/architecture as the target system

## Requirements

- Python >=3.10
- pyyaml >=6.0
- psutil >=5.9.0
- conda-pack >=0.7.0

## Acknowledgments

This project incorporates ideas and code from:
- [conda-execute](https://github.com/conda-tools/conda-execute) (BSD 3-Clause License)
- [conda-pack](https://github.com/conda/conda-pack) (BSD 3-Clause License)