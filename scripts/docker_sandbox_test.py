import os
import tempfile
import subprocess
import json
import shutil
from pathlib import Path

class DockerSandbox:
    def __init__(self):
        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(os.getcwd(), 'sandbox_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_python_code(self, code, requirements=None, timeout=300):
        """Run Python code in an isolated Docker container"""
        # Create a temporary directory for our files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write the Python script
            script_path = os.path.join(tmpdir, 'script.py')
            with open(script_path, 'w') as f:
                f.write(code)
            
            # Write requirements if provided
            if requirements:
                req_path = os.path.join(tmpdir, 'requirements.txt')
                with open(req_path, 'w') as f:
                    f.write('\n'.join(requirements))
            
            # Create Dockerfile
            dockerfile = f"""
FROM python:3.9-slim
WORKDIR /app
COPY script.py .
{f'COPY requirements.txt .' if requirements else ''}
{f'RUN pip install -r requirements.txt' if requirements else ''}
CMD ["python", "script.py"]
"""
            dockerfile_path = os.path.join(tmpdir, 'Dockerfile')
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile)
            
            try:
                # Build the image
                print("Building Docker image...")
                build_cmd = ['docker', 'build', '-q', tmpdir]
                image_id = subprocess.check_output(build_cmd).decode().strip()
                print("Image built successfully:", image_id)
                
                # Create a temporary output directory
                tmp_output = os.path.join(tmpdir, 'output')
                os.makedirs(tmp_output, exist_ok=True)
                
                # Run the container
                print("Starting container...")
                run_cmd = [
                    'docker', 'run',
                    '--rm',  # Remove container after it exits
                    '-v', f'{tmp_output}:/app/output',  # Mount output directory
                    image_id
                ]
                
                # Run the container and capture output
                result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=timeout)
                print("Container completed successfully")
                
                # Copy output files to permanent location
                for file in os.listdir(tmp_output):
                    src = os.path.join(tmp_output, file)
                    dst = os.path.join(self.output_dir, file)
                    shutil.copy2(src, dst)
                    print(f"Saved output file: {dst}")
                
                return result.stdout
            except subprocess.CalledProcessError as e:
                print(f"Error running container: {e}")
                if e.output:
                    print("Error output:", e.output.decode())
                return None
            except Exception as e:
                print(f"Unexpected error: {e}")
                return None

def test_basic_sandbox():
    """Test basic sandbox functionality"""
    print("\nRunning basic sandbox test...")
    sandbox = DockerSandbox()
    code = """
print('Hello from sandbox!')
"""
    result = sandbox.run_python_code(code)
    print("Basic test output:", result)

def test_image_analysis():
    """Test image analysis capabilities"""
    print("\nRunning image analysis test...")
    sandbox = DockerSandbox()
    requirements = [
        'numpy',
        'scikit-image',
        'matplotlib',
        'pillow'
    ]
    
    code = """
import numpy as np
from skimage import data
from skimage.filters import gaussian
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Load sample image
image = data.astronaut()
# Convert to grayscale
gray_image = rgb2gray(image)
# Apply Gaussian blur
blurred = gaussian(gray_image, sigma=3)

# Calculate some statistics
stats = {
    'mean': float(np.mean(gray_image)),
    'std': float(np.std(gray_image)),
    'shape': gray_image.shape
}

# Create and save a plot
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(image)
plt.title('Original')
plt.subplot(132)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale')
plt.subplot(133)
plt.imshow(blurred, cmap='gray')
plt.title('Blurred')
plt.savefig('/app/output/image_analysis.png')
plt.close()

print(f"Image statistics: {stats}")
"""
    
    result = sandbox.run_python_code(code, requirements)
    print("\nImage Analysis Results:")
    print(result)

def test_machine_learning():
    """Test machine learning capabilities"""
    print("\nRunning machine learning test...")
    sandbox = DockerSandbox()
    requirements = [
        'numpy',
        'scikit-learn',
        'matplotlib'
    ]
    
    code = """
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                         n_redundant=5, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Get predictions
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

# Feature importance plot
plt.figure(figsize=(10, 5))
plt.bar(range(20), clf.feature_importances_)
plt.title('Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.savefig('/app/output/feature_importance.png')
plt.close()

print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")
"""
    
    result = sandbox.run_python_code(code, requirements)
    print("\nMachine Learning Results:")
    print(result)

if __name__ == "__main__":
    print("Starting Docker Sandbox evaluation...")
    test_basic_sandbox()
    test_image_analysis()
    test_machine_learning()
    print("Evaluation complete!") 
