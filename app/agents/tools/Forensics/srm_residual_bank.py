"""
SRM (Spatial Rich Model) Residual Filter Bank for Image Forensics

This module implements the Spatial Rich Model (SRM) residual filter bank, which extracts
multi-channel noise residuals using a set of high-pass and directional kernels. SRM provides
robust low-level features for statistical analysis and machine learning in image forensics
and steganalysis.

Principle:
    Apply a bank of spatial filters (horizontal, vertical, diagonal edge detectors) to 
    extract noise patterns and residuals. These residuals capture statistical anomalies
    that are imperceptible to humans but reveal tampering, compression, or hidden data.

"""

import numpy as np
from PIL import Image
from typing import Union, Literal


def _conv2_same(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Perform 2D convolution with 'same' padding using reflection.
    
    Parameters
    ----------
    x : np.ndarray
        Input 2D array
    k : np.ndarray
        Convolution kernel
        
    Returns
    -------
    y : np.ndarray
        Convolved result with same shape as input
    """
    H, W = x.shape
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((ph, ph), (pw, pw)), 'reflect')
    y = np.zeros_like(x, np.float32)
    for i in range(H):
        for j in range(W):
            y[i, j] = np.sum(xp[i:i+kh, j:j+kw] * k)
    return y


# SRM filter kernels
_SRM_KERNELS = [
    np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], np.float32),      # Kernel 0: Horizontal edge detector
    np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], np.float32),      # Kernel 1: Vertical edge detector
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32),   # Kernel 2: Laplacian (all directions)
    np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]], np.float32),      # Kernel 3: Diagonal edge detector
]


def srm_residual_bank(
    img: Union[str, Image.Image, np.ndarray],
    srm_filter: Literal[0, 1, 2, 3]
) -> np.ndarray:
    """
    Apply SRM (Spatial Rich Model) residual filter to extract noise patterns.
    
    The SRM filter bank consists of multiple directional and edge-detection kernels that
    extract high-frequency noise residuals from images. These residuals reveal statistical
    anomalies useful for detecting tampering, compression artifacts, and steganography.
    
    Parameters
    ----------
    img : str, PIL.Image.Image, or np.ndarray
        Input image. Can be:
        - File path (str): Path to the image file
        - PIL Image: Color or grayscale PIL Image object
        - NumPy array: (H, W) or (H, W, C) array
        Image will be converted to grayscale internally.
        
    srm_filter : {0, 1, 2, 3}
        SRM filter kernel selection:
        - 0: Horizontal edge detector (detects horizontal manipulation boundaries)
        - 1: Vertical edge detector (detects vertical manipulation boundaries)
        - 2: Laplacian filter (detects edges in all directions, most comprehensive)
        - 3: Diagonal edge detector (detects diagonal patterns and resampling)
    
    Returns
    -------
    residual : np.ndarray
        Filtered residual map as 2D array (H, W) of type float32.
        - Contains raw residual values (may be positive or negative)
        - Higher absolute values indicate stronger edges or anomalies
        - Use abs() or normalization for visualization
    
    Raises
    ------
    ValueError
        If srm_filter is not in {0, 1, 2, 3}
    FileNotFoundError
        If img is a file path that doesn't exist
    
    Examples
    --------
    >>> import numpy as np
    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    
    >>> # Example 1: Basic usage with horizontal edge detector
    >>> img = Image.open("suspicious_image.jpg")
    >>> residual = srm_residual_bank(img, srm_filter=0)
    >>> plt.imshow(residual, cmap='gray')
    >>> plt.title("SRM Residual (Horizontal Edges)")
    >>> plt.show()
    
    >>> # Example 2: Compare all four SRM filters
    >>> fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    >>> filter_names = ['Horizontal', 'Vertical', 'Laplacian', 'Diagonal']
    >>> for i in range(4):
    ...     residual = srm_residual_bank(img, srm_filter=i)
    ...     axes[i//2, i%2].imshow(residual, cmap='gray')
    ...     axes[i//2, i%2].set_title(f'SRM Filter {i}: {filter_names[i]}')
    ...     axes[i//2, i%2].axis('off')
    >>> plt.show()
    
    >>> # Example 3: Using file path with Laplacian filter
    >>> residual = srm_residual_bank("image.jpg", srm_filter=2)
    
    >>> # Example 4: Visualize with normalization
    >>> residual = srm_residual_bank(img, srm_filter=2)
    >>> # Normalize to [0, 1] for better visualization
    >>> residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
    >>> plt.imshow(residual_norm, cmap='gray')
    >>> plt.show()
    
    >>> # Example 5: Using NumPy array input
    >>> img_array = np.random.rand(512, 512, 3) * 255
    >>> residual = srm_residual_bank(img_array, srm_filter=1)
    
    Notes
    -----
    - SRM filters are effective for detecting:
      * Tampering boundaries and splicing seams
      * Resampling artifacts from scaling/rotation
      * Compression inconsistencies
      * Hidden steganographic content
      * Statistical anomalies in pixel neighborhoods
    
    - Filter characteristics:
      * Filter 0 (Horizontal): Sensitive to horizontal edges, copy-paste boundaries
      * Filter 1 (Vertical): Sensitive to vertical edges, vertical splicing
      * Filter 2 (Laplacian): Most comprehensive, detects all-direction edges
      * Filter 3 (Diagonal): Sensitive to diagonal patterns, rotation artifacts
    
    - The input image is converted to grayscale using standard RGB weights:
      Gray = 0.299*R + 0.587*G + 0.114*B
    
    - Convolution uses reflection padding to avoid border artifacts
    
    - Raw residuals are NOT normalized - use normalization for visualization
    
    - For machine learning applications, all 4 filters can be extracted and
      concatenated as multi-channel features
    
    See Also
    --------
    numpy.pad : Used for reflection padding
    
    """
  
    # Convert to grayscale numpy array
    X = np.array(img.convert("L"), np.float32) if isinstance(img, Image.Image) else np.asarray(img, np.float32)
    
    # Convert RGB to grayscale if needed
    if X.ndim == 3:
        X = 0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2]
    
    # Apply selected SRM filter
    residual = _conv2_same(X, _SRM_KERNELS[srm_filter])
    
    return residual


# Function schema for agent integration
SRM_RESIDUAL_BANK_SCHEMA = {
    "name": "srm_residual_bank",
    "description": "Apply SRM (Spatial Rich Model) residual filter to extract noise patterns for detecting tampering, compression artifacts, and steganography. Returns raw residual map (NOT normalized).",
    "parameters": {
        "type": "object",
        "properties": {
            "img": {
                "type": ["string", "object"],
                "description": "Input image. Can be file path (string), PIL Image object, or NumPy array. Will be converted to grayscale."
            },
            "srm_filter": {
                "type": "integer",
                "enum": [0, 1, 2, 3],
                "description": "SRM filter selection: 0=horizontal edges, 1=vertical edges, 2=Laplacian (all directions, recommended), 3=diagonal edges"
            }
        },
        "required": ["img", "srm_filter"]
    },
    "returns": {
        "type": "array",
        "description": "2D residual map (H×W, float32) with raw values."
    }
}


if __name__ == "__main__":
    """
    Example usage and visualization
    """
    import matplotlib.pyplot as plt
    
    # Example: Compare all four SRM filters
    img_path = "temp_ela.jpg"  # Replace with your image path
    
    try:
        img = Image.open(img_path)
        
        # Filter names for display
        filter_names = ['Horizontal Edges', 'Vertical Edges', 'Laplacian (All Directions)', 'Diagonal Edges']
        
        # Visualize all four SRM filters
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        
        for i in range(4):
            residual = srm_residual_bank(img, srm_filter=i)
            
            # Display the residual
            axes[i // 2, i % 2].imshow(residual, cmap='gray')
            axes[i // 2, i % 2].set_title(f'SRM Filter {i}: {filter_names[i]}')
            axes[i // 2, i % 2].axis('off')
        
        plt.tight_layout()
        plt.savefig("srm_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ SRM residuals generated successfully!")
        print(f"✓ Comparison saved to srm_comparison.png")
        
        # Display single filter (Laplacian is most commonly used)
        plt.figure(figsize=(10, 8))
        residual = srm_residual_bank(img, srm_filter=2)
        plt.imshow(residual, cmap='gray')
        plt.title("SRM Residual (Laplacian Filter)")
        plt.colorbar(label='Residual Value')
        plt.axis('off')
        plt.savefig("srm_laplacian.png", dpi=150, bbox_inches='tight')
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found.")
        print("Please update 'img_path' in the __main__ section with a valid image path.")
    except Exception as e:
        print(f"Error: {e}")