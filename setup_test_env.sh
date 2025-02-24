#!/bin/bash

# Install test dependencies
pip install pytest pytest-cov numpy conda-pack psutil pyyaml

# Create test environments if they don't exist
if [ ! -f "env_with_numpy.tar.gz" ]; then
    # Create environment with numpy
    conda create -p ./test_env_numpy python=3.9 numpy -y
    conda-pack -p ./test_env_numpy -o env_with_numpy.tar.gz
    rm -rf ./test_env_numpy
fi

if [ ! -f "env_without_numpy.tar.gz" ]; then
    # Create environment without numpy
    conda create -p ./test_env_base python=3.9 -y
    conda-pack -p ./test_env_base -o env_without_numpy.tar.gz
    rm -rf ./test_env_base
fi

# Run tests
pytest conda_env_executor/tests/ -v --cov=conda_env_executor 