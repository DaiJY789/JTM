"""
DWT (Discrete Wavelet Transform) High-Frequency Subband for Image Forensics

This module implements multi-scale wavelet decomposition to extract high-frequency diagonal
detail (HH) subbands. DWT analysis reveals texture inconsistencies, synthetic patterns, and
upsampling artifacts at multiple scales, which are indicative of image manipulation.

Principle:
    Apply discrete wavelet transform to decompose the image into approximation (LL) and
    detail subbands (LH, HL, HH) at multiple levels. The HH (diagonal high-frequency) subband
    captures diagonal edges and texture details that are sensitive to manipulation, synthesis,
    and resampling artifacts.
"""

import numpy as np
import pywt
from PIL import Image
from typing import Union, Literal


def dwt_hh_map(
    img: Union[str, Image.Image, np.ndarray],
    level: Literal[1, 2, 3]
) -> np.ndarray:
    """
    Extract high-frequency diagonal (HH) subband using Discrete Wavelet Transform.
    
    This method applies multi-level DWT decomposition to extract the HH (diagonal high-frequency)
    subband, which captures diagonal edges, texture details, and high-frequency noise patterns.
    These features are particularly sensitive to manipulation artifacts, synthetic textures,
    and resampling operations.
    
    Parameters
    ----------
    img : str, PIL.Image.Image, or np.ndarray
        Input image. Can be:
        - File path (str): Path to the image file
        - PIL Image: RGB or grayscale PIL Image object
        - NumPy array: (H, W) or (H, W, C) array
        
    level : {1, 2, 3}
        Decomposition level for wavelet transform:
        - 1: Single-level decomposition (captures fine details, higher resolution)
        - 2: Two-level decomposition (captures medium-scale patterns)
        - 3: Three-level decomposition (captures coarse-scale structures)
        Higher levels analyze coarser features but at lower resolution.
    
    Returns
    -------
    hh_map : np.ndarray
        High-frequency diagonal (HH) subband as grayscale 2D array of type float32.
        - Shape: (H', W') where H', W' depend on decomposition level
        - Level 1: approximately (H/2, W/2)
        - Level 2: approximately (H/4, W/4)
        - Level 3: approximately (H/8, W/8)
        - Contains raw wavelet coefficients (may be positive or negative)
        - Higher absolute values indicate stronger diagonal features
    
    Raises
    ------
    ValueError
        If level is not in {1, 2, 3}
    FileNotFoundError
        If img is a file path that doesn't exist
    
    Examples
    --------
    >>> import numpy as np
    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    
    >>> # Example 1: Basic usage with level 1
    >>> img = Image.open("suspicious_image.jpg")
    >>> hh_map = dwt_hh_map(img, level=1)
    >>> plt.imshow(hh_map, cmap='gray')
    >>> plt.title("DWT HH Component (Level 1)")
    >>> plt.show()
    
    >>> # Example 2: Compare different decomposition levels
    >>> fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    >>> for idx, lv in enumerate([1, 2, 3]):
    ...     hh_map = dwt_hh_map(img, level=lv)
    ...     axes[idx].imshow(hh_map, cmap='gray')
    ...     axes[idx].set_title(f'DWT HH (Level {lv})')
    ...     axes[idx].axis('off')
    >>> plt.show()
    
    >>> # Example 3: Using file path
    >>> hh_map = dwt_hh_map("image.jpg", level=1)
    
    >>> # Example 4: Visualize with absolute values
    >>> hh_map = dwt_hh_map(img, level=1)
    >>> plt.imshow(np.abs(hh_map), cmap='hot')
    >>> plt.title("DWT HH Component (Absolute)")
    >>> plt.colorbar()
    >>> plt.show()
    
    >>> # Example 5: Enhance contrast for better visualization
    >>> hh_map = dwt_hh_map(img, level=1)
    >>> hh_normalized = (hh_map - hh_map.min()) / (hh_map.max() - hh_map.min() + 1e-8)
    >>> plt.imshow(hh_normalized, cmap='gray')
    >>> plt.show()
    
    Notes
    -----
    - DWT HH subband is effective for detecting:
      * Synthetic textures from GANs or diffusion models
      * Upsampling and downsampling artifacts
      * Splicing boundaries with texture inconsistencies
      * Copy-paste forgeries with different scales
      * Resampling traces from geometric transformations
    
    - Wavelet properties:
      * Uses 'bior1.3' (Biorthogonal 1.3) wavelet
      * Symmetric boundary extension to minimize edge artifacts
      * Multi-scale decomposition: level 1 (fine), 2 (medium), 3 (coarse)
    
    - Subband meanings:
      * HH: High-frequency diagonal details (extracted by this function)
      * HL: Horizontal details (not extracted)
      * LH: Vertical details (not extracted)
      * LL: Approximation/low-frequency (not extracted)
    
    - The function processes RGB channels independently and averages them:
      * Each channel undergoes DWT decomposition
      * HH subbands from R, G, B are averaged to produce grayscale output
      * Grayscale input is automatically converted to 3-channel
    
    - Resolution reduction per level:
      * Level 1: output is ~50% of input size
      * Level 2: output is ~25% of input size
      * Level 3: output is ~12.5% of input size
    
    - Raw coefficients are NOT normalized - consider normalization for visualization
    
    
    """
    # Validate level
    if level not in [1, 2, 3]:
        raise ValueError(f"level must be 1, 2, or 3, got {level}")
    
    # Load image if path is provided
    if isinstance(img, str):
        try:
            img = Image.open(img)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {img}")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    # Convert to RGB numpy array
    arr = np.array(img.convert("RGB"), dtype=np.float32) if isinstance(img, Image.Image) \
          else np.asarray(img, dtype=np.float32)
    
    # Convert grayscale to RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], -1)
    
    # Wavelet decomposition parameters
    wavelet = "bior1.3"  # Biorthogonal 1.3 wavelet
    
    # Extract HH subband for each RGB channel
    hh_channels = []
    for c in range(3):
        # Perform multi-level wavelet decomposition
        coeffs = pywt.wavedec2(arr[..., c], wavelet=wavelet, level=level, mode="symmetric")
        
        # Extract HH subband at the specified level
        # coeffs[0] is LL (approximation)
        # coeffs[1] to coeffs[level] are detail subbands (LH, HL, HH)
        # coeffs[level][2] is the HH subband at the specified level
        HH = coeffs[level][2].astype(np.float32)
        hh_channels.append(HH)
    
    # Average across RGB channels to get grayscale HH map
    hh_map = np.mean(hh_channels, axis=0)
    
    return hh_map


# Function schema for agent integration
DWT_HH_MAP_SCHEMA = {
    "name": "dwt_hh_map",
    "description": "Extract high-frequency diagonal (HH) subband using Discrete Wavelet Transform. Reveals texture inconsistencies, synthetic patterns, and upsampling artifacts at multiple scales.",
    "parameters": {
        "type": "object",
        "properties": {
            "img": {
                "type": ["string", "object"],
                "description": "Input image. Can be file path (string), PIL Image object, or NumPy array."
            },
            "level": {
                "type": "integer",
                "enum": [1, 2, 3],
                "description": "Decomposition level: 1=fine details (recommended), 2=medium-scale patterns, 3=coarse structures. Higher levels have lower resolution."
            }
        },
        "required": ["img", "level"]
    },
    "returns": {
        "type": "array",
        "description": "2D grayscale HH subband (float32) with raw wavelet coefficients. Size reduced by 2^level (level 1: ~50%, level 2: ~25%, level 3: ~12.5%)."
    }
}


if __name__ == "__main__":
    """
    Example usage and visualization
    """
    import matplotlib.pyplot as plt
    
    # Example: Compare different decomposition levels
    img_path = "temp_ela.jpg"  # Replace with your image path
    
    try:
        # Load image
        img = Image.open(img_path)
        
        # Compare different decomposition levels
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        levels = [1, 2, 3]
        
        for idx, lv in enumerate(levels):
            hh_map = dwt_hh_map(img, level=lv)
            
            # Display HH subband
            axes[idx].imshow(hh_map, cmap='gray')
            axes[idx].set_title(f'DWT HH Component (Level {lv})')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig("dwt_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ DWT HH subbands extracted successfully!")
        print(f"✓ Comparison saved to dwt_comparison.png")
        
        # Display single level with colorbar
        plt.figure(figsize=(10, 8))
        hh_map = dwt_hh_map(img, level=1)
        plt.imshow(hh_map, cmap='gray')
        plt.title("DWT HH Component (Level 1)")
        plt.colorbar(label='Wavelet Coefficient')
        plt.axis('off')
        plt.savefig("dwt_hh_level1.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Level 1 HH map shape: {hh_map.shape}")
        print(f"✓ Coefficient range: [{hh_map.min():.2f}, {hh_map.max():.2f}]")
        
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found.")
        print("Please update 'img_path' in the __main__ section with a valid image path.")
    except Exception as e:
        print(f"Error: {e}")