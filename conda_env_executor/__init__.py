"""
Conda Environment Executor - Execute Python code in isolated conda environments.

This package provides tools for executing Python code in isolated conda environments
with efficient data passing between environments using shared memory.

Copyright (c) 2024, Wei Ouyang and contributors.
Licensed under the BSD 3-Clause License.
"""

from .executor import CondaEnvExecutor, ExecutionResult, TimingInfo
from .env_spec import EnvSpec, read_env_spec
from .shared_memory import SharedMemoryChannel

__version__ = "0.1.0"
__all__ = [
    "CondaEnvExecutor",
    "ExecutionResult",
    "TimingInfo",
    "EnvSpec",
    "read_env_spec",
    "SharedMemoryChannel",
] 