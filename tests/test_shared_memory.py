"""Tests for the shared memory communication module."""

import numpy as np
import pytest
import os

from conda_env_executor.shared_memory import SharedMemoryChannel


def test_shared_memory_basic():
    """Test basic shared memory operations."""
    channel = SharedMemoryChannel()
    try:
        data = {"key": "value", "number": 42}
        channel.write_object(data)
        result = channel.read_object()
        assert result == data
    finally:
        channel.close()


def test_shared_memory_numpy():
    """Test shared memory operations with numpy arrays."""
    channel = SharedMemoryChannel()
    try:
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        channel.write_object(data)
        result = channel.read_object()
        assert np.array_equal(result, data)
        assert result.dtype == data.dtype
    finally:
        channel.close()


def test_shared_memory_large_data():
    """Test shared memory operations with large data."""
    size = 200 * 1024 * 1024  # 200MB
    channel = SharedMemoryChannel(size=size)
    try:
        data = np.random.rand(1000, 1000)  # ~8MB of data
        channel.write_object(data)
        result = channel.read_object()
        assert np.array_equal(result, data)
    finally:
        channel.close()


def test_shared_memory_too_large():
    """Test shared memory operations with data too large for the channel."""
    channel = SharedMemoryChannel(size=1024)  # 1KB
    try:
        data = np.random.rand(1000, 1000)  # ~8MB of data
        with pytest.raises(ValueError):
            channel.write_object(data)
    finally:
        channel.close()


def test_shared_memory_nested_structure():
    """Test shared memory operations with nested data structures."""
    channel = SharedMemoryChannel()
    try:
        data = {
            "array": np.array([1, 2, 3]),
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "number": 42
        }
        channel.write_object(data)
        result = channel.read_object()
        assert np.array_equal(result["array"], data["array"])
        assert result["list"] == data["list"]
        assert result["dict"] == data["dict"]
        assert result["number"] == data["number"]
    finally:
        channel.close()


def test_shared_memory_context_manager():
    """Test shared memory channel as context manager."""
    with SharedMemoryChannel() as channel:
        data = {"key": "value"}
        channel.write_object(data)
        result = channel.read_object()
        assert result == data


def test_shared_memory_multiple_writes():
    """Test multiple writes to shared memory channel."""
    with SharedMemoryChannel() as channel:
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}
        
        channel.write_object(data1)
        result1 = channel.read_object()
        assert result1 == data1
        
        channel.write_object(data2)
        result2 = channel.read_object()
        assert result2 == data2


def test_shared_memory_cleanup():
    """Test shared memory cleanup."""
    channel = SharedMemoryChannel()
    path = channel.shm_path
    channel.close()
    assert not os.path.exists(path)


def test_shared_memory_id():
    """Test shared memory channel ID."""
    channel_id = "test_channel"
    channel = SharedMemoryChannel(channel_id=channel_id)
    try:
        assert channel.id == channel_id
    finally:
        channel.close()


def test_shared_memory_different_numpy_dtypes():
    """Test shared memory operations with different numpy dtypes."""
    dtypes = [np.int32, np.int64, np.float32, np.float64, np.bool_]
    
    with SharedMemoryChannel() as channel:
        for dtype in dtypes:
            data = np.array([[1, 2], [3, 4]], dtype=dtype)
            channel.write_object(data)
            result = channel.read_object()
            assert np.array_equal(result, data)
            assert result.dtype == dtype


def test_shared_memory_large_array_compact():
    """Test shared memory operations with large arrays using compact representation."""
    with SharedMemoryChannel() as channel:
        # Create a large array that should trigger compact representation
        data = np.random.rand(2000, 2000)  # Much larger than the threshold
        channel.write_object(data)
        result = channel.read_object()
        assert np.array_equal(result, data)
        assert result.dtype == data.dtype
        assert result.shape == data.shape 