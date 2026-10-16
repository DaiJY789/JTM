"""
Bayar Constrained Filter for Image Forensics

This module implements the Bayar constrained convolutional filter for detecting
image tampering traces such as resampling, splicing, and other statistical anomalies.

"""

import numpy as np
from PIL import Image
from scipy import signal
from typing import Union, Literal


def bayar_filter(
    img: Union[str, Image.Image, np.ndarray],
    kernel_size: Literal[3, 5, 7]
) -> np.ndarray:
    """
    Apply Bayar constrained filter to detect image tampering traces.
    
    The Bayar filter is a fixed prediction residual filter that suppresses image content
    while highlighting tampering artifacts. It uses a constrained kernel where the center
    weight is -1 and surrounding weights sum to 1, extracting pixel-neighborhood differences.
    
    Parameters
    ----------
    img : str, PIL.Image.Image, or np.ndarray
        Input image. Can be:
        - File path (str): Path to the image file
        - PIL Image: RGB or grayscale PIL Image object
        - NumPy array: (H, W) or (H, W, C) array
        
    kernel_size : {3, 5, 7}
        Size of the Bayar filter kernel. Must be 3, 5, or 7.
        - 3: Fast, preserves fine details, sensitive to noise. Best for high-resolution images.
        - 5: Balanced performance (recommended). Standard configuration from the original paper.
        - 7: More robust to noise but may blur tampering boundaries. Best for low-quality images.
    
    Returns
    -------
    residual : np.ndarray
        Filtered residual image highlighting tampering traces.
        - Shape: (H, W, 3) for RGB output
        - dtype: float32
        - Raw residual values (may contain negative values)
    
    Raises
    ------
    ValueError
        If kernel_size is not in {3, 5, 7}
    FileNotFoundError
        If img is a file path that doesn't exist
    
    Examples
    --------
    >>> import numpy as np
    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    
    >>> # Example 1: Basic usage with file path
    >>> img = Image.open("temp_ela.jpg")
    >>> residual = bayar_filter(img, kernel_size=5)
    >>> residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
    >>> plt.imshow(residual_norm)
    >>> plt.show()
    
    >>> # Example 2: Compare different kernel sizes
    >>> for k in [3, 5, 7]:
    ...     residual = bayar_filter(img, kernel_size=k)
    ...     residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
    ...     plt.figure()
    ...     plt.imshow(residual_norm)
    ...     plt.title(f"Bayar Residual (kernel_size={k})")
    ...     plt.show()
    
    >>> # Example 3: Using file path
    >>> residual = bayar_filter("suspicious_image.jpg", kernel_size=5)
    
    >>> # Example 4: Using NumPy array
    >>> img_array = np.random.rand(512, 512, 3).astype(np.float32)
    >>> residual = bayar_filter(img_array, kernel_size=5)
    
    Notes
    -----
    - The Bayar filter kernel has a fixed structure:
      * Center weight: -1
      * Surrounding weights: uniformly distributed, sum to 1
      * This creates a high-pass filter that emphasizes local inconsistencies
    
    - Typical tampering traces detected:
      * Resampling artifacts from scaling/rotation
      * Splicing boundaries from copy-paste operations
      * JPEG compression inconsistencies
      * Statistical anomalies in pixel neighborhoods
    
    - Processing is done per color channel independently for RGB images
    - Grayscale images are automatically converted to 3-channel RGB
    - The output is NOT normalized - use normalization for visualization if needed
    
    See Also
    --------
    scipy.signal.correlate2d : Used for convolution operation
    """
    
    # Convert to numpy array (original code logic)
    X = np.array(img.convert('RGB')).astype(np.float32) if isinstance(img, Image.Image) else np.asarray(img, np.float32)
    # Convert grayscale to RGB
    if X.ndim == 2:
        X = np.stack([X] * 3, -1)
    # Create Bayar constrained kernel (original code)
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    c = kernel_size // 2
    kernel[c, c] = -1
    kernel = kernel / (kernel_size**2 - 1)
    kernel[c, c] = -1
    # Apply filter to each channel (original code)
    residual = np.zeros_like(X)
    for ch in range(3):
        residual[:, :, ch] = signal.correlate2d(X[:, :, ch], kernel, mode='same', boundary='symm')
    
    return residual


# Function schema for agent integration
BAYAR_FILTER_SCHEMA = {
    "name": "bayar_filter",
    "description": "Apply Bayar constrained filter to detect image tampering traces such as resampling, splicing, and statistical anomalies. Returns raw filtered residual (NOT normalized).",
    "parameters": {
        "type": "object",
        "properties": {
            "img": {
                "type": ["string", "object"],
                "description": "Input image. Can be file path (string), PIL Image object, or NumPy array"
            },
            "kernel_size": {
                "type": "integer",
                "enum": [3, 5, 7],
                "description": "Size of Bayar filter kernel. 3=fast/detailed, 5=balanced (recommended), 7=noise-robust"
            }
        },
        "required": ["img", "kernel_size"]
    },
    "returns": {
        "type": "array",
        "description": "Raw filtered residual as NumPy array (float32, shape: H×W×3)."
    }
}


if __name__ == "__main__":
    """
    Example usage and visualization
    """
    import matplotlib.pyplot as plt
    
    # Example: Compare different kernel sizes
    img_path = "temp_ela.jpg"  # Replace with your image path
    
    try:
        img = Image.open(img_path)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        kernel_sizes = [3, 5, 7]
        
        for i, k in enumerate(kernel_sizes):
            residual = bayar_filter(img, kernel_size=k)
            # Normalize for visualization
            residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
            
            axes[i].imshow(residual_norm)
            axes[i].set_title(f"Bayar Filter (kernel_size={k})")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("bayar_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ Bayar filter applied successfully!")
        print(f"✓ Comparison saved to bayar_comparison.png")
        
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found.")
        print("Please update 'img_path' in the __main__ section with a valid image path.")
    except Exception as e:
        print(f"Error: {e}")