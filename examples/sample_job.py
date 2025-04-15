"""
Sample job for testing the conda environment executor.

This code contains a function named 'execute' that will be called by the executor.
The function takes an input and returns an output, which can be any JSON-serializable object.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def execute(input_data):
    """
    Process the input data and return results.
    
    This function is called by the conda environment executor.
    
    Args:
        input_data: The input data passed to the executor.
                   Should be a dictionary with at least a 'n_samples' key.
    
    Returns:
        A dictionary with results and visualizations.
    """
    print("Starting execution of sample job...")
    
    # Default values
    n_samples = 1000
    sleep_time = 2
    
    # Process input data if available
    if input_data and isinstance(input_data, dict):
        n_samples = input_data.get('n_samples', n_samples)
        sleep_time = input_data.get('sleep_time', sleep_time)
    
    print(f"Generating {n_samples} samples and sleeping for {sleep_time} seconds...")
    
    # Simulate CPU-intensive operation
    data = np.random.normal(0, 1, size=n_samples)
    df = pd.DataFrame({'values': data})
    stats = {
        'mean': float(df['values'].mean()),
        'median': float(df['values'].median()),
        'std': float(df['values'].std()),
        'min': float(df['values'].min()),
        'max': float(df['values'].max()),
    }
    
    # Simulate time-consuming operation
    time.sleep(sleep_time)
    
    # Create a visualization
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7)
    plt.title("Histogram of Random Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    # Save plot to a base64-encoded string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    # Return results
    return {
        'statistics': stats,
        'sample_size': n_samples,
        'execution_time': sleep_time,
        'histogram_plot': plot_data
    } 