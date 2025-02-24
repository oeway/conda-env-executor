# Conda Environment Executor

A Python package for executing code in isolated conda environments with efficient data passing between environments.

## Features

- Execute Python code in isolated conda environments
- Efficient data passing between environments using shared memory
- Support for conda-pack environments
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

### Basic Usage

```python
from conda_env_executor import CondaEnvExecutor

# Define your code
numpy_code = '''
import numpy as np
from typing import Any, Dict

def execute(input_data: Any) -> Dict:
    arr = np.array(input_data)
    return {
        "mean": arr.mean().item(),
        "std": arr.std().item(),
        "shape": list(arr.shape)
    }
'''

# Execute in a conda environment
with CondaEnvExecutor('env_with_numpy.tar.gz') as executor:
    result = executor.execute(numpy_code, input_data=[[1, 2, 3], [4, 5, 6]])
    print(result.result)  # {"mean": 3.5, "std": 1.7078...}
```

### Using Environment Specifications

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

```python
from conda_env_executor import CondaEnvExecutor

executor = CondaEnvExecutor.from_yaml('environment.yaml')
with executor:
    result = executor.execute(code, input_data)
```

### Temporary Environments

```python
from conda_env_executor import CondaEnvExecutor

# Create a temporary environment with specific packages
executor = CondaEnvExecutor.create_temp_env(['python=3.11', 'numpy', 'scipy'])
with executor:
    result = executor.execute(code, input_data)
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