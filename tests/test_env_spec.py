"""Tests for the environment specification module."""

import os
import tempfile
from pathlib import Path
import tarfile
import io

import pytest
import yaml

from conda_env_executor.env_spec import EnvSpec, read_env_spec, extract_spec_from_code


def test_env_spec_defaults():
    """Test EnvSpec default values."""
    spec = EnvSpec()
    assert spec.name is None
    assert spec.channels == ['conda-forge']
    assert spec.dependencies == []
    assert spec.prefix is None


def test_env_spec_from_yaml(tmp_path):
    """Test creating EnvSpec from YAML file."""
    yaml_content = """
    name: test-env
    channels:
      - conda-forge
      - defaults
    dependencies:
      - python=3.9
      - numpy
    """
    yaml_file = tmp_path / "env.yaml"
    yaml_file.write_text(yaml_content)
    
    spec = EnvSpec.from_yaml(yaml_file)
    assert spec.name == "test-env"
    assert spec.channels == ["conda-forge", "defaults"]
    assert spec.dependencies == ["python=3.9", "numpy"]


def test_env_spec_from_dict():
    """Test creating EnvSpec from dictionary."""
    spec_dict = {
        "name": "test-env",
        "channels": ["conda-forge", "defaults"],
        "dependencies": ["python=3.9", "numpy"],
        "prefix": "/path/to/env"
    }
    
    spec = EnvSpec.from_dict(spec_dict)
    assert spec.name == "test-env"
    assert spec.channels == ["conda-forge", "defaults"]
    assert spec.dependencies == ["python=3.9", "numpy"]
    assert spec.prefix == "/path/to/env"


def test_extract_spec_from_code():
    """Test extracting environment specification from code comments."""
    code = '''
# conda env
# channels:
#   - conda-forge
#   - defaults
# dependencies:
#   - python=3.9
#   - numpy

def execute(input_data):
    return input_data
'''
    
    spec = extract_spec_from_code(code)
    assert spec is not None
    assert spec.channels == ["conda-forge", "defaults"]
    assert spec.dependencies == ["python=3.9", "numpy"]


def test_extract_spec_from_code_no_spec():
    """Test extracting environment specification from code without spec."""
    code = '''
def execute(input_data):
    return input_data
'''
    
    spec = extract_spec_from_code(code)
    assert spec is None


def test_read_env_spec_from_yaml_file(tmp_path):
    """Test reading environment specification from YAML file."""
    yaml_content = """
    name: test-env
    channels:
      - conda-forge
    dependencies:
      - python=3.9
    """
    yaml_file = tmp_path / "env.yaml"
    yaml_file.write_text(yaml_content)
    
    spec = read_env_spec(yaml_file)
    assert spec.name == "test-env"
    assert spec.channels == ["conda-forge"]
    assert spec.dependencies == ["python=3.9"]


def test_read_env_spec_from_yaml_string():
    """Test reading environment specification from YAML string."""
    yaml_str = """
    name: test-env
    channels:
      - conda-forge
    dependencies:
      - python=3.9
    """
    
    spec = read_env_spec(yaml_str)
    assert spec.name == "test-env"
    assert spec.channels == ["conda-forge"]
    assert spec.dependencies == ["python=3.9"]


def test_read_env_spec_from_dict():
    """Test reading environment specification from dictionary."""
    spec_dict = {
        "name": "test-env",
        "channels": ["conda-forge"],
        "dependencies": ["python=3.9"]
    }
    
    spec = read_env_spec(spec_dict)
    assert spec.name == "test-env"
    assert spec.channels == ["conda-forge"]
    assert spec.dependencies == ["python=3.9"]


def test_read_env_spec_from_code():
    """Test reading environment specification from code string."""
    code = '''
# conda env
# channels:
#   - conda-forge
# dependencies:
#   - python=3.9

def execute(input_data):
    return input_data
'''
    
    spec = read_env_spec(code)
    assert spec.channels == ["conda-forge"]
    assert spec.dependencies == ["python=3.9"]


def test_read_env_spec_invalid_yaml():
    """Test reading environment specification from invalid YAML."""
    with pytest.raises(ValueError, match="Could not parse environment specification from string"):
        read_env_spec("invalid: yaml: content: - not valid")


def test_read_env_spec_nonexistent_file():
    """Test reading environment specification from nonexistent file."""
    with pytest.raises(FileNotFoundError):
        read_env_spec("/nonexistent/file.yaml")


def test_read_env_spec_invalid_type():
    """Test reading environment specification from invalid type."""
    with pytest.raises(TypeError):
        read_env_spec(123)


def test_read_env_spec_conda_pack_file(tmp_path):
    """Test reading environment specification from conda-pack file."""
    # Create a proper tar.gz file
    pack_file = tmp_path / "env.tar.gz"
    with tarfile.open(pack_file, 'w:gz') as tar:
        # Add a dummy file to make it a valid tar.gz
        info = tarfile.TarInfo(name="dummy.txt")
        info.size = len(b"dummy content")
        tar.addfile(info, io.BytesIO(b"dummy content"))

    # Reading a conda-pack file should return a default EnvSpec
    spec = read_env_spec(pack_file)
    assert isinstance(spec, EnvSpec)
    assert spec.name is None
    assert spec.channels == []
    assert spec.dependencies == []


def test_read_env_spec_invalid_yaml_file(tmp_path):
    """Test reading environment specification from invalid YAML file."""
    # Create an invalid YAML file
    yaml_file = tmp_path / "invalid.yaml"
    yaml_file.write_text("invalid: - yaml: content")
    
    with pytest.raises(ValueError, match="Could not parse environment file"):
        read_env_spec(yaml_file) 