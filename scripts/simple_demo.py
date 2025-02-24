from env_executor import CondaEnvExecutor, ExecutionResult
import time

def print_result(name: str, result: ExecutionResult) -> None:
    """Pretty print execution results"""
    print(f"\n=== {name} Results ===")
    print(f"Success: {result.success}")
    
    if result.success:
        print("\nOutput:")
        print(result.result)
    else:
        print("\nError:")
        print(result.error)
    
    if result.stdout:
        print("\nStdout:")
        print(result.stdout)
    
    if result.stderr:
        print("\nStderr:")
        print(result.stderr)
    
    if result.timing:
        print("\nTiming:")
        print(f"  Environment setup: {result.timing.env_setup:.2f} seconds")
        print(f"  Code execution: {result.timing.code_execution:.2f} seconds")
        print(f"  Total time: {result.timing.env_setup + result.timing.code_execution:.2f} seconds")

def main():
    # Test data
    input_data = [[1, 2, 3], [4, 5, 6]]
    
    # Define the test code
    numpy_code = '''
import numpy as np
from typing import Any, Dict

def execute(input_data: Any) -> Dict:
    """Process input data using NumPy."""
    # Convert input to numpy array
    arr = np.array(input_data)
    
    # Perform some numpy operations
    result = {
        "mean": arr.mean().item(),
        "std": arr.std().item(),
        "sum": arr.sum().item(),
        "shape": list(arr.shape),
        "processed": (arr * 2).tolist()  # Double all values
    }
    
    return result
'''

    basic_code = '''
from typing import Any, Dict
import statistics
from itertools import chain

def execute(input_data: Any) -> Dict:
    """Process input data using basic Python."""
    # Flatten the list for statistical calculations
    flat_data = list(chain.from_iterable(input_data))
    
    # Calculate statistics using basic Python
    result = {
        "mean": statistics.mean(flat_data),
        "std": statistics.stdev(flat_data),
        "sum": sum(flat_data),
        "shape": [len(input_data), len(input_data[0])],
        "processed": [[x * 2 for x in row] for row in input_data]  # Double all values
    }
    
    return result
'''
    
    print("Testing execution in different conda environments with caching...")
    
    # First run - environments will be extracted
    print("\nFirst run (cold start)...")
    
    print("\nTesting NumPy environment...")
    with CondaEnvExecutor('env_with_numpy.tar.gz') as executor:
        result = executor.execute(numpy_code, input_data)
        print_result("NumPy Script", result)
    
    print("\nTesting basic environment...")
    with CondaEnvExecutor('env_without_numpy.tar.gz') as executor:
        result = executor.execute(basic_code, input_data)
        print_result("Basic Script", result)
    
    # Second run - should use cached environments
    print("\nSecond run (using cached environments)...")
    
    print("\nTesting NumPy environment...")
    with CondaEnvExecutor('env_with_numpy.tar.gz') as executor:
        result = executor.execute(numpy_code, input_data)
        print_result("NumPy Script", result)
    
    print("\nTesting basic environment...")
    with CondaEnvExecutor('env_without_numpy.tar.gz') as executor:
        result = executor.execute(basic_code, input_data)
        print_result("Basic Script", result)

if __name__ == '__main__':
    main() 
