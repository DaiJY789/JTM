"""
Laplacian High-Pass Filter for Image Forensics

This module implements a Laplacian-based high-pass filter to extract high-frequency
components from images. By computing second-order derivatives, this method emphasizes
regions with rapid intensity changes, revealing edges, textures, and manipulation artifacts
such as splicing boundaries, resampling traces, and tampering evidence.

Principle:
    Apply a Laplacian kernel to compute second-order derivatives, which emphasizes regions
    where grayscale values change rapidly. The Laplacian operator is rotation-invariant and
    detects edges in all directions simultaneously, making it effective for revealing
    high-frequency manipulation artifacts.
"""

import numpy as np
from PIL import Image
from scipy import signal
from typing import Union


def highpass_filter(
    img: Union[str, Image.Image, np.ndarray]
) -> np.ndarray:
    """
    Apply Laplacian high-pass filter to extract high-frequency components.
    
    The Laplacian operator computes second-order derivatives to detect regions with rapid
    intensity changes. This high-pass filter emphasizes edges, textures, and fine details
    while suppressing smooth regions, making it effective for revealing manipulation
    artifacts in forensic analysis.
    
    Parameters
    ----------
    img : str, PIL.Image.Image, or np.ndarray
        Input image. Can be:
        - File path (str): Path to the image file
        - PIL Image: RGB or grayscale PIL Image object
        - NumPy array: (H, W) or (H, W, C) array
        Image will be converted to grayscale internally.
    
    Returns
    -------
    high_freq : np.ndarray
        High-frequency residual as 2D grayscale array of type float32.
        - Shape: (H, W) same as input image
        - Values represent second-order derivatives (curvature)
        - Positive values: Local maxima (peaks)
        - Negative values: Local minima (valleys)
        - Raw values (not normalized), typically in range [-1000, 1000]
    
    Raises
    ------
    FileNotFoundError
        If img is a file path that doesn't exist
    ValueError
        If image cannot be loaded or processed
    
    Examples
    --------
    >>> import numpy as np
    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    
    >>> # Example 1: Basic high-pass filtering
    >>> img = Image.open("image.jpg")
    >>> high_freq = highpass_filter(img)
    >>> high_freq_norm = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min() + 1e-8)
    >>> plt.imshow(high_freq_norm, cmap='gray')
    >>> plt.title("Laplacian High-Pass Filter")
    >>> plt.show()
    
    >>> # Example 2: Using file path
    >>> high_freq = highpass_filter("suspicious_image.jpg")
    
    >>> # Example 3: Compare original and high-frequency
    >>> fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    >>> axes[0].imshow(img, cmap='gray')
    >>> axes[0].set_title('Original Image')
    >>> axes[1].imshow(high_freq_norm, cmap='gray')
    >>> axes[1].set_title('High-Frequency Components')
    >>> plt.show()
    
    >>> # Example 4: Visualize with absolute values
    >>> high_freq_abs = np.abs(high_freq)
    >>> plt.imshow(high_freq_abs, cmap='hot')
    >>> plt.title("High-Frequency (Absolute)")
    >>> plt.colorbar()
    >>> plt.show()
    
    >>> # Example 5: Using NumPy array
    >>> img_array = np.random.rand(512, 512) * 255
    >>> high_freq = highpass_filter(img_array)
    
    Notes
    -----
    - Laplacian high-pass filtering is effective for detecting:
      * Splicing and copy-paste boundaries
      * Resampling artifacts from geometric transformations
      * Double JPEG compression traces
      * Inpainting and cloning artifacts
      * Edge tampering and blending inconsistencies
      * Texture anomalies in forged regions
    
    - Technical details:
      * Laplacian kernel (4-connected):
        [[ 0, -1,  0],
         [-1,  4, -1],
         [ 0, -1,  0]]
      * Computes: ∂²I/∂x² + ∂²I/∂y² (second derivative approximation)
      * Center weight: 4 (sum of all neighbors)
      * Neighbor weights: -1 (subtract neighborhood)
      * Uses symmetric boundary padding
    
    - Laplacian operator properties:
      * Rotation-invariant (detects edges in all directions)
      * Approximates second-order derivative
      * Zero-crossing indicates edge location
      * Magnitude indicates edge strength
      * Sensitive to noise (use with caution on noisy images)
    
    - Why high-pass filtering reveals tampering:
      * Natural images have smooth intensity variations
      * Manipulation introduces discontinuities
      * Splicing creates artificial edges
      * Resampling introduces periodic high-frequency patterns
      * Different processing histories leave different high-frequency signatures
    
    - Interpretation in forensics:
      * Uniform high-frequency pattern: Likely authentic
      * Localized high-frequency anomalies: Potential tampering
      * Abrupt changes at boundaries: Splicing or copy-paste
      * Asymmetric patterns: Different processing applied to regions
      * Periodic structures: Resampling or interpolation artifacts
    
    - Advantages:
      * Simple and computationally efficient
      * Rotation-invariant (no directional bias)
      * Effective for general-purpose edge detection
      * Well-established in forensic literature
    
    - Limitations:
      * Very sensitive to noise
      * May amplify JPEG compression artifacts
      * Does not distinguish between natural and artificial edges
      * Requires careful interpretation in noisy images
    
    - The input image is converted to grayscale using standard weights:
      Gray = 0.299*R + 0.587*G + 0.114*B
    
    - Raw residuals are NOT normalized - use normalization for visualization:
```python
      high_freq_norm = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min() + 1e-8)
```
    
    - Alternative visualizations:
      * Absolute values: Shows edge strength regardless of direction
      * Hot colormap: Better highlights strong responses
      * Thresholding: Isolates strongest high-frequency components
    
    See Also
    --------
    scipy.signal.correlate2d : 2D convolution for filtering
    
    """
    # Load image if path is provided
    if isinstance(img, str):
        try:
            img = Image.open(img)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {img}")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    # Convert to grayscale numpy array
    if isinstance(img, Image.Image):
        X = np.array(img.convert('L')).astype(np.float32)
    else:
        X = np.asarray(img, np.float32)
        # Convert RGB to grayscale if needed
        if X.ndim == 3:
            X = 0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2]
    
    # Define Laplacian kernel (4-connected)
    kernel = np.array([[ 0, -1,  0],
                       [-1,  4, -1],
                       [ 0, -1,  0]], np.float32)
    
    # Apply Laplacian filter
    high_freq = signal.correlate2d(X, kernel, mode='same', boundary='symm')
    
    return high_freq


# Function schema for agent integration
HIGHPASS_FILTER_SCHEMA = {
    "name": "highpass_filter",
    "description": "Apply Laplacian high-pass filter to extract high-frequency components, revealing edges and manipulation artifacts. Effective for detecting splicing, resampling, and tampering through second-order derivative analysis.",
    "parameters": {
        "type": "object",
        "properties": {
            "img": {
                "type": ["string", "object"],
                "description": "Input image. Can be file path (string), PIL Image object, or NumPy array. Will be converted to grayscale."
            }
        },
        "required": ["img"]
    },
    "returns": {
        "type": "array",
        "description": "2D high-frequency residual (H×W, float32) with raw second-order derivative values."
    }
}


if __name__ == "__main__":
    """
    Example usage and visualization
    """
    import matplotlib.pyplot as plt
    
    img_path = "temp_ela.jpg"  # Replace with your image path
    
    try:
        # Load image and apply high-pass filter
        img = Image.open(img_path)
        high_freq = highpass_filter(img)
        
        # Normalize for visualization
        high_freq_norm = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min() + 1e-8)
        
        # Display
        plt.figure(figsize=(10, 8))
        plt.imshow(high_freq_norm, cmap='gray')
        plt.title("High-pass Filter (Laplacian)")
        plt.axis('off')
        # plt.show()
        plt.savefig("highpass_filter.png", dpi=150, bbox_inches='tight')
        
        print(f"✓ High-pass filtering complete")
        print(f"✓ High-frequency range: [{high_freq.min():.2f}, {high_freq.max():.2f}]")
        
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")