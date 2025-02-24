from sandboxai import Sandbox
import matplotlib.pyplot as plt
import numpy as np

def test_basic_sandbox():
    """Test basic sandbox functionality"""
    with Sandbox(embedded=True) as box:
        # Test basic Python execution
        result = box.run_ipython_cell("print('Hello from sandbox!')")
        print("Basic test output:", result.output)
        
        # Test pip installation
        result = box.run_shell_command("pip install numpy scikit-learn scikit-image matplotlib")
        print("Package installation output:", result.output)

def test_image_analysis():
    """Test image analysis capabilities"""
    code = """
import numpy as np
from skimage import data
from skimage.filters import gaussian
from skimage.color import rgb2gray

# Load sample image
image = data.astronaut()
# Convert to grayscale
gray_image = rgb2gray(image)
# Apply Gaussian blur
blurred = gaussian(gray_image, sigma=3)

# Calculate some statistics
stats = {
    'mean': np.mean(gray_image),
    'std': np.std(gray_image),
    'shape': gray_image.shape
}

# Create and save a plot
import matplotlib.pyplot as plt
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
plt.savefig('image_analysis.png')
plt.close()

print(f"Image statistics: {stats}")
"""
    
    with Sandbox(embedded=True) as box:
        result = box.run_ipython_cell(code)
        print("\nImage Analysis Results:")
        print(result.output)

def test_machine_learning():
    """Test machine learning capabilities"""
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
plt.savefig('feature_importance.png')
plt.close()

print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")
"""
    
    with Sandbox(embedded=True) as box:
        result = box.run_ipython_cell(code)
        print("\nMachine Learning Results:")
        print(result.output)

if __name__ == "__main__":
    print("Starting SandboxAI evaluation...")
    test_basic_sandbox()
    test_image_analysis()
    test_machine_learning()
    print("Evaluation complete!") 