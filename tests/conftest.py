"""Pytest configuration for conda_env_executor tests."""

import os
import subprocess
import pytest


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test that requires conda"
    )


@pytest.fixture(scope="session")
def conda_env_with_numpy(tmp_path_factory):
    """Create a conda environment with numpy for testing."""
    if os.path.exists("env_with_numpy.tar.gz"):
        return "env_with_numpy.tar.gz"
    
    # Create a temporary environment
    env_dir = tmp_path_factory.mktemp("conda_env")
    env_yaml = env_dir / "environment.yaml"
    env_yaml.write_text("""
    name: test_env
    channels:
      - conda-forge
    dependencies:
      - python=3.9
      - numpy
    """)
    
    # Create environment using conda-pack
    subprocess.run(
        ["conda-pack", "-p", str(env_dir), "-o", "env_with_numpy.tar.gz"],
        check=True
    )
    
    return "env_with_numpy.tar.gz"


@pytest.fixture(scope="session")
def conda_env_without_numpy(tmp_path_factory):
    """Create a conda environment without numpy for testing."""
    if os.path.exists("env_without_numpy.tar.gz"):
        return "env_without_numpy.tar.gz"
    
    # Create a temporary environment
    env_dir = tmp_path_factory.mktemp("conda_env")
    env_yaml = env_dir / "environment.yaml"
    env_yaml.write_text("""
    name: test_env
    channels:
      - conda-forge
    dependencies:
      - python=3.9
    """)
    
    # Create environment using conda-pack
    subprocess.run(
        ["conda-pack", "-p", str(env_dir), "-o", "env_without_numpy.tar.gz"],
        check=True
    )
    
    return "env_without_numpy.tar.gz" 