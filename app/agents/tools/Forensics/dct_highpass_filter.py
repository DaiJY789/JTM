"""
DCT-Based High-Pass Filter for Image Forensics

This module implements DCT (Discrete Cosine Transform) based high-pass filtering to extract
high-frequency components from images. By suppressing low-frequency content in the DCT domain,
this method highlights edges, textures, and manipulation artifacts such as splicing boundaries,
resampling traces, and JPEG compression inconsistencies.

Principle:
    Transform the image to DCT frequency domain, apply a circular high-pass mask centered at
    DC component to suppress low frequencies, then transform back to spatial domain. The
    resulting high-frequency residual emphasizes fine details and artifacts introduced by
    image manipulation operations.

"""

import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from typing import Union


def _dct2(x: np.ndarray) -> np.ndarray:
    """Compute 2D DCT (Discrete Cosine Transform)."""
    return dct(dct(x.T, norm='ortho').T, norm='ortho')


def _idct2(x: np.ndarray) -> np.ndarray:
    """Compute 2D inverse DCT."""
    return idct(idct(x.T, norm='ortho').T, norm='ortho')


def dct_highpass_filter(
    img: Union[str, Image.Image, np.ndarray]
) -> np.ndarray:
    """
    Apply DCT-based high-pass filter to extract high-frequency components.
    
    This method transforms the image to the DCT frequency domain, applies a circular
    high-pass mask to suppress low-frequency components (DC and nearby coefficients),
    and transforms back to spatial domain. The resulting high-frequency residual
    emphasizes edges, textures, and manipulation artifacts.
    
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
        - Contains raw high-frequency values (may be positive or negative)
        - Higher absolute values indicate stronger high-frequency content
        - Emphasizes edges, textures, and manipulation artifacts
    
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
    
    >>> # Example 1: Basic usage
    >>> img = Image.open("suspicious_image.jpg")
    >>> high_freq = dct_highpass_filter(img)
    >>> plt.imshow(high_freq, cmap='gray')
    >>> plt.title("DCT High-Pass Filter")
    >>> plt.colorbar()
    >>> plt.show()
    
    >>> # Example 2: Enhance visualization with normalization
    >>> high_freq = dct_highpass_filter(img)
    >>> high_freq_norm = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min() + 1e-8)
    >>> plt.imshow(high_freq_norm, cmap='gray')
    >>> plt.show()
    
    >>> # Example 3: Using file path
    >>> high_freq = dct_highpass_filter("image.jpg")
    
    >>> # Example 4: Compare original and high-frequency
    >>> fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    >>> axes[0].imshow(img, cmap='gray')
    >>> axes[0].set_title('Original Image')
    >>> axes[1].imshow(np.abs(high_freq), cmap='hot')
    >>> axes[1].set_title('High-Frequency Components')
    >>> plt.show()
    
    >>> # Example 5: Using NumPy array
    >>> img_array = np.random.rand(512, 512) * 255
    >>> high_freq = dct_highpass_filter(img_array)
    
    Notes
    -----
    - DCT high-pass filtering is effective for detecting:
      * Splicing and copy-paste boundaries
      * Resampling artifacts from scaling/rotation
      * JPEG compression inconsistencies
      * Double JPEG compression traces
      * Edge tampering and blending operations
      * Texture inconsistencies in forged regions
    
    - Technical details:
      * Uses 2D DCT (Discrete Cosine Transform)
      * Applies circular high-pass mask with radius_ratio = 0.3
      * Suppresses central 30% of frequency spectrum (low frequencies)
      * Preserves outer 70% (high frequencies)
      * Mask center is at DC component (top-left in DCT domain)
      * Distance measured from DC component in frequency space
    
    - DCT properties:
      * Energy compaction: most energy in low frequencies
      * DC coefficient: average value of the block
      * Low frequencies: smooth variations, overall structure
      * High frequencies: edges, textures, fine details, artifacts
    
    - High-pass mask design:
```
      mask = 1 if distance_from_DC > 0.3 * max_distance else 0
```
      This creates a circular mask suppressing the central 30% radius
    
    - Advantages over spatial high-pass filters:
      * Better frequency selectivity (sharp cutoff)
      * No ringing artifacts (unlike ideal filters in Fourier domain)
      * Natural for JPEG-related forensics (JPEG uses DCT)
      * Computational efficiency with fast DCT algorithms
    
    - Interpretation:
      * Uniform high-frequency pattern: Likely authentic
      * Localized high-frequency anomalies: Potential tampering
      * Abrupt changes at boundaries: Splicing or copy-paste
      * Periodic patterns: Resampling or grid-based manipulation
    
    - The input image is converted to grayscale using standard weights:
      Gray = 0.299*R + 0.587*G + 0.114*B
    
    - Raw residuals are NOT normalized - consider normalization for visualization:
```python
      high_freq_norm = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min() + 1e-8)
```
    
    - The radius_ratio parameter is fixed at 0.3 (suppress central 30%)
      * Smaller values (e.g., 0.2): More aggressive, keeps more frequencies
      * Larger values (e.g., 0.4): More conservative, removes more frequencies
    
    
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
    
    # Get image dimensions
    H, W = X.shape
    
    # Apply 2D DCT
    dct_coeffs = _dct2(X)
    
    # Create circular high-pass mask
    radius_ratio = 0.3  # Suppress central 30% of frequency spectrum
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    max_dist = np.sqrt(cy**2 + cx**2)
    
    # Mask: 1 for high frequencies (distance > threshold), 0 for low frequencies
    mask = (dist > max_dist * radius_ratio).astype(np.float32)
    
    # Apply mask to DCT coefficients
    dct_filtered = dct_coeffs * mask
    
    # Inverse DCT to get high-frequency residual
    high_freq = _idct2(dct_filtered)
    
    return high_freq


# Function schema for agent integration
DCT_HIGHPASS_FILTER_SCHEMA = {
    "name": "dct_highpass_filter",
    "description": "Apply DCT-based high-pass filter to extract high-frequency components, highlighting edges, textures, and manipulation artifacts. Effective for detecting splicing, resampling, and JPEG compression inconsistencies.",
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
        "description": "2D high-frequency residual (H×W, float32) with raw values."
    }
}


if __name__ == "__main__":
    """
    Example usage and visualization
    """
    import matplotlib.pyplot as plt
    
    # Example: Apply DCT high-pass filter and visualize
    img_path = "temp_ela.jpg"  # Replace with your image path
    
    try:
        # Load image
        img = Image.open(img_path)
        
        # Apply DCT high-pass filter
        high_freq = dct_highpass_filter(img)
        
        # Normalize for better visualization
        high_freq_norm = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min() + 1e-8)
        
        # Visualize original and high-frequency side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Original image
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # High-frequency components
        im = axes[1].imshow(high_freq_norm, cmap='gray')
        axes[1].set_title('DCT High-Pass Filter (Normalized)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Normalized Intensity')
        
        plt.tight_layout()
        plt.savefig("dct_highpass_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("=" * 60)
        print("DCT High-Pass Filter Results")
        print("=" * 60)
        print(f"✓ High-frequency residual extracted successfully!")
        print(f"✓ Comparison saved to dct_highpass_comparison.png")
        print(f"✓ Residual shape: {high_freq.shape}")
        print(f"✓ Raw value range: [{high_freq.min():.2f}, {high_freq.max():.2f}]")
        print(f"✓ Mean: {high_freq.mean():.2f}, Std: {high_freq.std():.2f}")
        
        # Display high-frequency alone with different colormap
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Grayscale
        axes[0].imshow(high_freq_norm, cmap='gray')
        axes[0].set_title('High-Frequency (Grayscale)')
        axes[0].axis('off')
        
        # Hot colormap for better artifact visibility
        im = axes[1].imshow(high_freq_norm, cmap='hot')
        axes[1].set_title('High-Frequency (Hot Colormap)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Intensity')
        
        plt.tight_layout()
        plt.savefig("dct_highpass_colormaps.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Colormap comparison saved to dct_highpass_colormaps.png")
        
        
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found.")
        print("Please update 'img_path' in the __main__ section with a valid image path.")
    except Exception as e:
        print(f"Error: {e}")