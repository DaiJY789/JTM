"""
Sobel Edge Detection for Image Forensics

This module implements the Sobel edge detection operator to detect edges and gradients in
images. Sobel filtering is widely used in image forensics to identify splicing boundaries,
inconsistent edge patterns, and manipulation artifacts by revealing abrupt intensity changes
that may indicate tampering.

Principle:
    Apply 3×3 convolution kernels to compute horizontal and vertical gradients using first-order
    derivatives. The Sobel operator emphasizes regions with rapid grayscale changes, revealing
    edges that may be natural or artificially introduced by manipulation operations.

"""

import numpy as np
from PIL import Image
from scipy import signal
from typing import Union


def sobel_filter(
    img: Union[str, Image.Image, np.ndarray]
) -> np.ndarray:
    """
    Apply Sobel edge detection to compute gradient magnitude.
    
    The Sobel operator detects edges by computing horizontal and vertical gradients using
    3×3 convolution kernels. The gradient magnitude reveals regions with rapid intensity
    changes, which can indicate natural edges or manipulation boundaries in forensic analysis.
    
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
    magnitude : np.ndarray
        Edge magnitude map as 2D grayscale array of type float32.
        - Shape: (H, W) same as input image
        - Values represent gradient magnitude (edge strength)
        - Higher values indicate stronger edges
        - Raw values (not normalized), typically in range [0, 1000+]
    
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
    
    >>> # Example 1: Basic edge detection
    >>> img = Image.open("image.jpg")
    >>> edges = sobel_filter(img)
    >>> edges_norm = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    >>> plt.imshow(edges_norm, cmap='gray')
    >>> plt.title("Sobel Edge Detection")
    >>> plt.show()
    
    >>> # Example 2: Using file path
    >>> edges = sobel_filter("suspicious_image.jpg")
    
    >>> # Example 3: Compare original and edges
    >>> fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    >>> axes[0].imshow(img, cmap='gray')
    >>> axes[0].set_title('Original Image')
    >>> axes[1].imshow(edges_norm, cmap='gray')
    >>> axes[1].set_title('Sobel Edges')
    >>> plt.show()
    
    >>> # Example 4: Threshold for strong edges
    >>> edges = sobel_filter(img)
    >>> threshold = np.percentile(edges, 90)  # Top 10% strongest edges
    >>> strong_edges = edges > threshold
    >>> plt.imshow(strong_edges, cmap='binary')
    >>> plt.title("Strong Edges (Top 10%)")
    >>> plt.show()
    
    >>> # Example 5: Using NumPy array
    >>> img_array = np.random.rand(512, 512) * 255
    >>> edges = sobel_filter(img_array)
    
    Notes
    -----
    - Sobel edge detection is effective for:
      * Identifying splicing and copy-paste boundaries
      * Detecting inconsistent edge patterns across regions
      * Revealing double JPEG compression artifacts at edges
      * Highlighting resampling-induced edge distortions
      * Finding unnatural edges from inpainting or cloning
    
    - Technical details:
      * Horizontal Sobel kernel (Gx):
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]
      * Vertical Sobel kernel (Gy):
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]
      * Gradient magnitude: sqrt(Gx² + Gy²)
      * Uses symmetric boundary padding to avoid edge artifacts
    
    - Sobel operator properties:
      * Approximates first-order derivative
      * Emphasizes edges at multiple orientations
      * Provides both edge magnitude and direction
      * More robust to noise than simple gradient operators
      * Central difference with smoothing (weighted average)
    
    - Edge interpretation in forensics:
      * Natural edges: Smooth, consistent gradient patterns
      * Splicing boundaries: Abrupt, unnatural edge transitions
      * Resampling artifacts: Jagged or aliased edges
      * Double JPEG: Blocky edge patterns (8×8 grid)
      * Cloning/inpainting: Missing or artificial edge continuity
    
    - Advantages over other edge detectors:
      * Simple and computationally efficient
      * Good balance of edge detection and noise suppression
      * Well-established baseline for forensic analysis
      * Easier to interpret than more complex methods
    
    - Limitations:
      * Sensitive to noise in low-quality images
      * May miss edges in low-contrast regions
      * Does not provide edge direction explicitly (only magnitude)
      * Less sophisticated than Canny or learning-based methods
    
    - The input image is converted to grayscale using standard weights:
      Gray = 0.299*R + 0.587*G + 0.114*B
    
    - Raw magnitudes are NOT normalized - use normalization for visualization:
```python
      edges_norm = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
```
    
    - For forensic analysis, compare edge patterns between:
      * Suspected manipulated regions vs. authentic regions
      * Different compression quality levels
      * Boundaries vs. interior regions
    
    See Also
    --------
    scipy.signal.correlate2d : 2D convolution for gradient computation
  
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
    
    # Define Sobel kernels
    # Horizontal gradient kernel (detects vertical edges)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], np.float32)
    
    # Vertical gradient kernel (detects horizontal edges)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], np.float32)
    
    # Compute horizontal and vertical gradients
    grad_x = signal.correlate2d(X, sobel_x, mode='same', boundary='symm')
    grad_y = signal.correlate2d(X, sobel_y, mode='same', boundary='symm')
    
    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)
    
    return magnitude


# Function schema for agent integration
SOBEL_FILTER_SCHEMA = {
    "name": "sobel_filter",
    "description": "Apply Sobel edge detection to compute gradient magnitude, revealing edges and boundaries. Effective for identifying splicing boundaries, inconsistent edge patterns, and manipulation artifacts in forensic analysis.",
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
        "description": "2D edge magnitude map (H×W, float32) with raw gradient values. Higher values indicate stronger edges."
    }
}


if __name__ == "__main__":
    """
    Example usage and visualization
    """
    import matplotlib.pyplot as plt
    
    img_path = "temp_ela.jpg"  # Replace with your image path
    
    try:
        # Load image and apply Sobel filter
        img = Image.open(img_path)
        edges = sobel_filter(img)
        
        # Normalize for visualization
        edges_norm = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
        
        # Display
        plt.figure(figsize=(10, 8))
        plt.imshow(edges_norm, cmap='gray')
        plt.title("Sobel Edge Detection")
        plt.axis('off')
        # plt.show()
        plt.savefig("sobel_filter.png", dpi=150, bbox_inches='tight')
        
        print(f"✓ Edge detection complete")
        print(f"✓ Edge range: [{edges.min():.2f}, {edges.max():.2f}]")
        
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")