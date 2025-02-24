import os
import sys
import json
import tempfile
import subprocess
import mmap
import struct
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Union, NamedTuple
from dataclasses import dataclass
import uuid
import textwrap
import shutil

class TimingInfo(NamedTuple):
    """Timing information for environment operations"""
    env_setup: float  # Time spent setting up the environment
    code_execution: float  # Time spent executing the code

@dataclass
class ExecutionResult:
    """Holds the result of executing a function in a conda environment"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    timing: Optional[TimingInfo] = None

def compute_file_hash(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

class EnvCache:
    """Manages cached conda environments"""
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.conda_env_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_cache_index()
    
    def _load_cache_index(self) -> None:
        """Load or create the cache index file"""
        self.index_path = os.path.join(self.cache_dir, "cache_index.json")
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                self.cache_index = json.load(f)
        else:
            self.cache_index = {}
    
    def _save_cache_index(self) -> None:
        """Save the cache index file"""
        with open(self.index_path, 'w') as f:
            json.dump(self.cache_index, f)
    
    def get_env_path(self, env_pack_path: str) -> Optional[str]:
        """
        Get the path to a cached environment if it exists and is valid.
        
        Args:
            env_pack_path: Path to the conda-pack file
            
        Returns:
            Path to the cached environment or None if not found/invalid
        """
        try:
            file_hash = compute_file_hash(env_pack_path)
            if file_hash in self.cache_index:
                cache_info = self.cache_index[file_hash]
                cache_path = cache_info['path']
                
                # Verify the cache exists and is valid
                if os.path.exists(cache_path) and os.path.isdir(cache_path):
                    # Check if the activation script exists
                    if os.path.exists(os.path.join(cache_path, 'bin', 'activate')):
                        return cache_path
            
            return None
        except Exception:
            return None
    
    def add_env(self, env_pack_path: str, env_path: str) -> None:
        """
        Add an environment to the cache.
        
        Args:
            env_pack_path: Path to the conda-pack file
            env_path: Path to the extracted environment
        """
        file_hash = compute_file_hash(env_pack_path)
        self.cache_index[file_hash] = {
            'path': env_path,
            'pack_path': str(Path(env_pack_path).resolve()),
            'created_at': time.time()
        }
        self._save_cache_index()
    
    def cleanup_old_envs(self, max_age_days: int = 30) -> None:
        """
        Remove environments older than max_age_days.
        
        Args:
            max_age_days: Maximum age in days for cached environments
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        to_remove = []
        for file_hash, cache_info in self.cache_index.items():
            if current_time - cache_info['created_at'] > max_age_seconds:
                env_path = cache_info['path']
                if os.path.exists(env_path):
                    try:
                        shutil.rmtree(env_path)
                    except Exception:
                        pass
                to_remove.append(file_hash)
        
        for file_hash in to_remove:
            del self.cache_index[file_hash]
        
        if to_remove:
            self._save_cache_index()

class SharedMemoryChannel:
    """Handles shared memory communication between processes"""
    def __init__(self, channel_id: Optional[str] = None, size: int = 1024*1024):
        self.size = size
        self.channel_id = channel_id or str(uuid.uuid4())
        self.shm_path = f'/tmp/shm_{self.channel_id}'
        
        # Create a temporary file for shared memory
        self.fd = os.open(self.shm_path, os.O_CREAT | os.O_RDWR, 0o666)
        os.truncate(self.fd, size)
        # Memory map the file
        self.memory = mmap.mmap(self.fd, size)
        
    def write_object(self, obj: Any) -> None:
        """Serialize and write a Python object to shared memory"""
        msg = json.dumps(obj)
        msg_bytes = msg.encode('utf-8')
        size = len(msg_bytes)
        if size + 4 > self.size:  # 4 bytes for size header
            raise ValueError(f"Object too large for shared memory channel (size: {size}, max: {self.size-4})")
        self.memory.seek(0)
        self.memory.write(struct.pack('I', size) + msg_bytes)
        
    def read_object(self) -> Any:
        """Read and deserialize a Python object from shared memory"""
        self.memory.seek(0)
        size = struct.unpack('I', self.memory.read(4))[0]
        msg_bytes = self.memory.read(size)
        return json.loads(msg_bytes.decode('utf-8'))
        
    def close(self) -> None:
        """Clean up shared memory resources"""
        self.memory.close()
        os.close(self.fd)
        try:
            os.unlink(self.shm_path)
        except:
            pass

    @property
    def id(self) -> str:
        return self.channel_id

class CondaEnvExecutor:
    """Executes Python functions in a specified conda environment"""
    def __init__(self, env_pack_path: str, cache_dir: Optional[str] = None):
        self.env_pack_path = env_pack_path
        self._temp_dir = None
        self._env_dir = None
        self._is_extracted = False
        self._owns_env_dir = False  # Whether we created the env dir or using cached
        self.env_cache = EnvCache(cache_dir)
        
    def _extract_env(self) -> float:
        """
        Extract the conda-packed environment
        
        Returns:
            Time taken to set up the environment in seconds
        """
        if self._is_extracted:
            return 0.0
            
        start_time = time.time()
        
        # Check if we have a cached version
        cached_env = self.env_cache.get_env_path(self.env_pack_path)
        if cached_env:
            self._env_dir = cached_env
            self._is_extracted = True
            self._owns_env_dir = False
            return time.time() - start_time
            
        # Create a new environment directory in the cache
        cache_subdir = str(uuid.uuid4())
        self._env_dir = os.path.join(self.env_cache.cache_dir, cache_subdir)
        self._owns_env_dir = True
        
        os.makedirs(self._env_dir, exist_ok=True)
        subprocess.run(
            ['tar', 'xf', self.env_pack_path, '-C', self._env_dir], 
            check=True
        )
        # Fix the activation script for the new location
        subprocess.run(
            ['bash', '-c', f'source {self._env_dir}/bin/activate && conda-unpack'],
            check=True, 
            cwd=self._env_dir
        )
        
        # Add to cache
        self.env_cache.add_env(self.env_pack_path, self._env_dir)
        self._is_extracted = True
        
        # Cleanup old environments
        self.env_cache.cleanup_old_envs()
        
        return time.time() - start_time

    def execute(self, code: str, input_data: Any = None) -> ExecutionResult:
        """
        Execute Python code in the conda environment.
        The code should define an 'execute' function that takes a single argument.
        
        Args:
            code: String containing Python code with an execute(input_data) function
            input_data: Data to pass to the execute function
        
        Returns:
            ExecutionResult containing success status and output/error information
        """
        try:
            # Time the environment setup
            env_setup_time = self._extract_env()
            
            # Time the code execution
            exec_start_time = time.time()
            
            # Create shared memory channel for communication
            channel = SharedMemoryChannel()
            
            if input_data is not None:
                channel.write_object(input_data)
            
            # Create the execution wrapper code
            wrapper_code = f'''
import os
import json
import mmap
import struct
import sys
import traceback

def setup_shared_memory():
    fd = os.open('{channel.shm_path}', os.O_RDWR)
    return mmap.mmap(fd, {channel.size})

def read_input(memory):
    memory.seek(0)
    size = struct.unpack('I', memory.read(4))[0]
    msg_bytes = memory.read(size)
    return json.loads(msg_bytes.decode('utf-8'))

def write_output(memory, obj):
    msg = json.dumps(obj)
    msg_bytes = msg.encode('utf-8')
    size = len(msg_bytes)
    memory.seek(0)
    memory.write(struct.pack('I', size) + msg_bytes)

try:
    # Execute the user code to define the execute function
{textwrap.indent(code.strip(), '    ')}
    
    # Setup shared memory
    memory = setup_shared_memory()
    
    try:
        # Read input if any
        input_data = read_input(memory) if {input_data is not None} else None
        
        # Execute the function
        result = execute(input_data)
        
        # Write result
        write_output(memory, {{"success": True, "result": result}})
        
    finally:
        memory.close()
        
except Exception as e:
    error_msg = {{"success": False, "error": str(e), "traceback": traceback.format_exc()}}
    try:
        memory = setup_shared_memory()
        write_output(memory, error_msg)
        memory.close()
    except:
        print(json.dumps(error_msg))
        raise
'''
            
            # Run the code in the conda environment
            activate_cmd = f'source {self._env_dir}/bin/activate'
            escaped_code = wrapper_code.replace('"', '\\"').replace('$', '\\$')
            python_cmd = f'python -c "{escaped_code}"'
            full_cmd = f'{activate_cmd} && {python_cmd}'
            
            try:
                result = subprocess.run(
                    ['bash', '-c', full_cmd],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Read the result from shared memory
                output = channel.read_object()
                
                # Calculate execution time
                exec_time = time.time() - exec_start_time
                timing = TimingInfo(env_setup=env_setup_time, code_execution=exec_time)
                
                if output['success']:
                    return ExecutionResult(
                        success=True,
                        result=output['result'],
                        stdout=result.stdout,
                        stderr=result.stderr,
                        timing=timing
                    )
                else:
                    return ExecutionResult(
                        success=False,
                        error=output['error'],
                        stdout=result.stdout,
                        stderr=result.stderr,
                        timing=timing
                    )
                    
            except subprocess.CalledProcessError as e:
                exec_time = time.time() - exec_start_time
                timing = TimingInfo(env_setup=env_setup_time, code_execution=exec_time)
                return ExecutionResult(
                    success=False,
                    error=str(e),
                    stdout=e.stdout,
                    stderr=e.stderr,
                    timing=timing
                )
                
        finally:
            channel.close()
    
    def cleanup(self) -> None:
        """Clean up temporary files"""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
        if self._owns_env_dir and self._env_dir and os.path.exists(self._env_dir):
            shutil.rmtree(self._env_dir)
        self._is_extracted = False
        self._temp_dir = None
        self._env_dir = None
        self._owns_env_dir = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 
