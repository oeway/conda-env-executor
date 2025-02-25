"""Tests for the main executor module."""

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from conda_env_executor.executor import (
    CondaEnvExecutor,
    EnvCache,
    ExecutionResult,
    TimingInfo,
    compute_file_hash
)


def create_test_env_yaml(tmp_path):
    """Create a test environment YAML file."""
    yaml_content = """name: test-env
channels:
  - conda-forge
dependencies:
  - python=3.9
  - numpy"""
    yaml_file = tmp_path / "env.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file


def test_compute_file_hash(tmp_path):
    """Test file hash computation."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    
    hash1 = compute_file_hash(test_file)
    hash2 = compute_file_hash(test_file)
    assert hash1 == hash2
    
    test_file.write_text("different content")
    hash3 = compute_file_hash(test_file)
    assert hash1 != hash3


def test_env_cache(tmp_path):
    """Test environment cache operations."""
    cache_dir = tmp_path / "cache"
    cache = EnvCache(str(cache_dir))
    
    # Create a test environment file
    env_file = tmp_path / "env.tar.gz"
    env_file.write_bytes(b"test content")
    
    # Test adding environment to cache
    env_path = str(tmp_path / "env")
    os.makedirs(env_path)
    os.makedirs(os.path.join(env_path, "bin"))
    with open(os.path.join(env_path, "bin", "activate"), "w") as f:
        f.write("# activation script")
    
    cache.add_env(str(env_file), env_path)
    
    # Test retrieving environment from cache
    cached_path = cache.get_env_path(str(env_file))
    assert cached_path == env_path
    
    # Test cleanup of old environments
    cache.cleanup_old_envs(max_age_days=0)
    assert cache.get_env_path(str(env_file)) is None


@pytest.mark.integration
def test_executor_with_yaml(tmp_path):
    """Test executor with YAML environment specification."""
    yaml_file = create_test_env_yaml(tmp_path)
    
    executor = CondaEnvExecutor.from_yaml(yaml_file)
    try:
        code = """
        import numpy as np
        
        def execute(input_data):
            return np.mean(input_data)
        """
        
        result = executor.execute(code, input_data=[1, 2, 3, 4])
        assert result.success
        assert result.result == 2.5
        assert isinstance(result.timing, TimingInfo)
    finally:
        executor.cleanup()


@pytest.mark.integration
def test_executor_with_conda_pack(tmp_path):
    """Test executor with conda-pack environment."""
    # This test requires a pre-built conda-pack environment
    if not os.path.exists("env_with_numpy.tar.gz"):
        pytest.skip("Conda-pack environment not available")
    
    executor = CondaEnvExecutor("env_with_numpy.tar.gz")
    try:
        code = """
        import numpy as np
        
        def execute(input_data):
            return np.mean(input_data)
        """
        
        result = executor.execute(code, input_data=[1, 2, 3, 4])
        assert result.success
        assert result.result == 2.5
    finally:
        executor.cleanup()


@pytest.mark.integration
def test_executor_temp_env():
    """Test executor with temporary environment."""
    executor = CondaEnvExecutor.create_temp_env(
        packages=["python=3.9", "numpy"],
        channels=["conda-forge"]
    )
    try:
        code = """
        import numpy as np
        
        def execute(input_data):
            return np.mean(input_data)
        """
        
        result = executor.execute(code, input_data=[1, 2, 3, 4])
        assert result.success
        assert result.result == 2.5
    finally:
        executor.cleanup()


def test_executor_error_handling():
    """Test executor error handling."""
    executor = CondaEnvExecutor.create_temp_env(["python=3.9"])
    try:
        code = """
        def execute(input_data):
            raise ValueError("Test error")
        """
        
        result = executor.execute(code)
        assert not result.success
        assert "Test error" in result.error
    finally:
        executor.cleanup()


def test_executor_numpy_array():
    """Test executor with numpy array input/output."""
    executor = CondaEnvExecutor.create_temp_env(["python=3.9", "numpy"])
    try:
        code = """
        import numpy as np
        
        def execute(input_data):
            return {
                'mean': np.mean(input_data),
                'std': np.std(input_data),
                'shape': input_data.shape
            }
        """
        
        data = np.array([[1, 2], [3, 4]])
        result = executor.execute(code, input_data=data)
        assert result.success
        assert result.result['mean'] == 2.5
        assert result.result['shape'] == (2, 2)
    finally:
        executor.cleanup()


def test_executor_context_manager():
    """Test executor as context manager."""
    with CondaEnvExecutor.create_temp_env(["python=3.9"]) as executor:
        code = """
        def execute(input_data):
            return input_data
        """
        
        result = executor.execute(code, input_data=42)
        assert result.success
        assert result.result == 42


def test_executor_cleanup():
    """Test executor cleanup."""
    executor = CondaEnvExecutor.create_temp_env(["python=3.9"])
    executor.cleanup()
    assert executor._env_dir is None
    assert executor._env_path is None
    assert not executor._is_extracted


@pytest.mark.integration
def test_executor_large_data():
    """Test executor with large data transfer."""
    executor = CondaEnvExecutor.create_temp_env(["python=3.9", "numpy"])
    try:
        code = """
        import numpy as np
        
        def execute(input_data):
            return np.mean(input_data)
        """
        
        data = np.random.rand(1000, 1000)
        result = executor.execute(code, input_data=data)
        assert result.success
        assert np.isclose(result.result, np.mean(data))
    finally:
        executor.cleanup()


@pytest.mark.integration
def test_executor_multiple_executions():
    """Test multiple executions with same executor."""
    executor = CondaEnvExecutor.create_temp_env(["python=3.9"])
    try:
        code = """
        def execute(input_data):
            return input_data * 2
        """
        
        for i in range(5):
            result = executor.execute(code, input_data=i)
            assert result.success
            assert result.result == i * 2
    finally:
        executor.cleanup()


def test_executor_invalid_code():
    """Test executor with invalid Python code."""
    executor = CondaEnvExecutor.create_temp_env(["python=3.9"])
    try:
        code = """
        def execute(input_data)
            return input_data  # Missing colon
        """
        
        result = executor.execute(code)
        assert not result.success
        assert "SyntaxError" in result.error
    finally:
        executor.cleanup()


def test_executor_missing_execute_function():
    """Test executor with code missing execute function."""
    executor = CondaEnvExecutor.create_temp_env(["python=3.9"])
    try:
        code = """
        def wrong_name(input_data):
            return input_data
        """
        
        result = executor.execute(code)
        assert not result.success
        assert "NameError" in result.error
    finally:
        executor.cleanup() 