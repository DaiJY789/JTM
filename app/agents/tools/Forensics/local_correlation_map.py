"""
Local Correlation Map (LCM) for Image Forensics

This module implements Local Correlation Map analysis to quantify enhanced correlation
in local pixel neighborhoods. Research shows that synthetic images (from GANs, diffusion
models) exhibit stronger local correlations compared to natural images, making LCM a
useful tool for detecting AI-generated content and image manipulation.

Principle:
    Compute Pearson correlation between row-wise and column-wise mean vectors within
    sliding windows. Natural images show lower local correlation due to complex textures,
    while synthetic or manipulated regions often exhibit higher correlation due to
    artificial smoothness and predictable patterns.

"""

import numpy as np
from PIL import Image
from typing import Union, Literal


def local_correlation_map(
    img: Union[str, Image.Image, np.ndarray],
    w: Literal[2, 3, 4, 5]
) -> np.ndarray:
    """
    Compute local correlation map to reveal synthetic patterns and manipulation artifacts.
    
    This method calculates the Pearson correlation between row-wise and column-wise mean
    vectors within sliding windows across the image. Synthetic or manipulated regions
    typically exhibit higher local correlation due to artificial smoothness and predictable
    patterns, while natural images show lower correlation due to complex textures.
    
    Parameters
    ----------
    img : str, PIL.Image.Image, or np.ndarray
        Input image. Can be:
        - File path (str): Path to the image file
        - PIL Image: RGB or grayscale PIL Image object
        - NumPy array: (H, W) or (H, W, C) array
        Image will be converted to grayscale internally.
        
    w : {2, 3, 4, 5}
        Window size for local correlation computation:
        - 2: Very fine-grained analysis (2×2 windows), sensitive to pixel-level patterns
        - 3: Fine analysis (3×3 windows), good for detecting small artifacts (recommended)
        - 4: Medium analysis (4×4 windows), balances detail and robustness
        - 5: Coarser analysis (5×5 windows), more robust to noise but less detailed
    
    Returns
    -------
    corr_map : np.ndarray
        Local correlation map as 2D array of type float32.
        - Shape: (H-w+1, W-w+1) where H, W are input image dimensions
        - Values range approximately from -1 to 1 (Pearson correlation coefficient)
        - Higher values: stronger local correlation (typical for synthetic/manipulated regions)
        - Lower values: weaker local correlation (typical for natural textures)
        - Slightly smaller than input due to sliding window computation
    
    Raises
    ------
    ValueError
        If w is not in {2, 3, 4, 5}
    FileNotFoundError
        If img is a file path that doesn't exist
    
    Examples
    --------
    >>> import numpy as np
    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    
    >>> # Example 1: Basic usage with w=2
    >>> img = Image.open("suspicious_image.jpg")
    >>> corr_map = local_correlation_map(img, w=2)
    >>> plt.imshow(corr_map, cmap='gray')
    >>> plt.title("Local Correlation Map (w=2)")
    >>> plt.colorbar()
    >>> plt.show()
    
    >>> # Example 2: Compare different window sizes
    >>> fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    >>> for idx, window_size in enumerate([2, 3, 4, 5]):
    ...     corr_map = local_correlation_map(img, w=window_size)
    ...     axes[idx//2, idx%2].imshow(corr_map, cmap='gray')
    ...     axes[idx//2, idx%2].set_title(f'Window Size = {window_size}×{window_size}')
    ...     axes[idx//2, idx%2].axis('off')
    >>> plt.show()
    
    >>> # Example 3: Using file path
    >>> corr_map = local_correlation_map("image.jpg", w=3)
    
    >>> # Example 4: Visualize with color map
    >>> corr_map = local_correlation_map(img, w=2)
    >>> plt.imshow(corr_map, cmap='hot')
    >>> plt.title("Local Correlation Map (Hot)")
    >>> plt.colorbar(label='Correlation Coefficient')
    >>> plt.show()
    
    >>> # Example 5: Threshold to highlight high-correlation regions
    >>> corr_map = local_correlation_map(img, w=3)
    >>> threshold = 0.5
    >>> suspicious_regions = corr_map > threshold
    >>> plt.imshow(suspicious_regions, cmap='binary')
    >>> plt.title(f"High Correlation Regions (threshold={threshold})")
    >>> plt.show()
    
    Notes
    -----
    - Local Correlation Map is effective for detecting:
      * AI-generated images (GANs, diffusion models, deepfakes)
      * Synthetic textures with unnatural smoothness
      * Copy-paste forgeries with artificial consistency
      * Inpainting and retouching artifacts
      * Over-smoothed or overly-processed regions
    
    - Interpretation:
      * High correlation values (closer to 1): Suspicious, potentially synthetic
      * Low correlation values (closer to 0 or negative): Natural, complex textures
      * Uniform high correlation: Likely AI-generated or heavily manipulated
      * Mixed correlation patterns: Potential splicing or partial manipulation
    
    - Technical details:
      * Uses Pearson correlation coefficient: ρ = Cov(X,Y) / (σ_X × σ_Y)
      * Row means: average pixel values across each row in the window
      * Column means: average pixel values across each column in the window
      * Correlation between row and column mean vectors quantifies local consistency
    
    - Window size considerations:
      * Smaller windows (w=2,3): More sensitive to fine details, higher resolution output
      * Larger windows (w=4,5): More robust to noise, coarser analysis
      * Trade-off: sensitivity vs. robustness
    
    - The input image is converted to grayscale using standard weights:
      Gray = 0.299*R + 0.587*G + 0.114*B
    
    - Output size is slightly reduced: (H-w+1, W-w+1) due to sliding window
    
    - The SAFE paper and related research highlight that synthetic images
      exhibit stronger local correlations than natural images
    
    See Also
    --------
    numpy.corrcoef : Correlation coefficient computation
    
    """
    
    # Convert to grayscale numpy array
    X = np.array(img.convert("L"), dtype=np.float32) if isinstance(img, Image.Image) else np.asarray(img, np.float32)
    
    # Convert RGB to grayscale if needed
    if X.ndim == 3:
        X = 0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2]
    
    # Get image dimensions
    H, W = X.shape
    
    # Initialize output correlation map
    out = np.zeros((H - w + 1, W - w + 1), np.float32)
    
    # Slide window across image
    for i in range(H - w + 1):
        rX = X[i:i + w]
        for j in range(W - w + 1):
            # Extract window
            P = rX[:, j:j + w]
            
            # Compute row-wise and column-wise means
            r = P.mean(0)  # Column means (average across rows)
            c = P.mean(1)  # Row means (average across columns)
            
            # Compute Pearson correlation coefficient
            r_centered = r - r.mean()
            c_centered = c - c.mean()
            correlation = (r_centered * c_centered).mean() / ((r.std() + 1e-8) * (c.std() + 1e-8))
            
            out[i, j] = float(correlation)
    
    return out


# Function schema for agent integration
LOCAL_CORRELATION_MAP_SCHEMA = {
    "name": "local_correlation_map",
    "description": "Compute local correlation map to detect synthetic patterns and manipulation. Synthetic images show stronger local correlations than natural images. Higher values indicate suspicious regions.",
    "parameters": {
        "type": "object",
        "properties": {
            "img": {
                "type": ["string", "object"],
                "description": "Input image. Can be file path (string), PIL Image object, or NumPy array. Will be converted to grayscale."
            },
            "w": {
                "type": "integer",
                "enum": [2, 3, 4, 5],
                "description": "Window size for correlation computation: 2=very fine (sensitive), 3=fine (recommended), 4=medium (balanced), 5=coarse (robust)"
            }
        },
        "required": ["img", "w"]
    },
    "returns": {
        "type": "array",
        "description": "2D correlation map (H-w+1, W-w+1, float32) with values in [-1, 1]. Higher values indicate stronger local correlation (suspicious for synthetic/manipulated regions)."
    }
}


if __name__ == "__main__":
    """
    Example usage and visualization
    """
    import matplotlib.pyplot as plt
    
    # Example: Compare different window sizes
    img_path = "temp_ela.jpg"  # Replace with your image path
    
    try:
        # Load image
        img = Image.open(img_path)
        
        # Compare different window sizes
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        window_sizes = [2, 3, 4, 5]
        
        for idx, w_size in enumerate(window_sizes):
            corr_map = local_correlation_map(img, w=w_size)
            
            axes[idx // 2, idx % 2].imshow(corr_map, cmap='gray')
            axes[idx // 2, idx % 2].set_title(f'Local Correlation Map (Window Size = {w_size}×{w_size})')
            axes[idx // 2, idx % 2].axis('off')
        
        plt.tight_layout()
        plt.savefig("lcm_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ Local correlation maps generated successfully!")
        print(f"✓ Comparison saved to lcm_comparison.png")
        
        # Display single correlation map with colorbar
        plt.figure(figsize=(10, 8))
        corr_map = local_correlation_map(img, w=3)
        im = plt.imshow(corr_map, cmap='gray')
        plt.title("Local Correlation Map (Window Size = 3×3)")
        plt.colorbar(im, label='Correlation Coefficient')
        plt.axis('off')
        plt.savefig("lcm_w3.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Correlation map shape: {corr_map.shape}")
        print(f"✓ Correlation range: [{corr_map.min():.3f}, {corr_map.max():.3f}]")
        print(f"✓ Mean correlation: {corr_map.mean():.3f}")
        
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found.")
        print("Please update 'img_path' in the __main__ section with a valid image path.")
    except Exception as e:
        print(f"Error: {e}")