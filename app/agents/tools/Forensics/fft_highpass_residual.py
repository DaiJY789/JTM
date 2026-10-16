"""
FFT High-Pass Residual for Image Forensics

This module implements FFT-based high-pass filtering to emphasize high-frequency artifacts
in images. By removing low-frequency components in the frequency domain, this method reveals
forgery boundaries, texture anomalies, and manipulation traces that are independent of JPEG
compression history.

Principle:
    Transform image to frequency domain using FFT, remove low-frequency components by masking
    the center region, then transform back to spatial domain. High-frequency residuals reveal
    subtle manipulation artifacts, splicing boundaries, and synthetic patterns.
"""

import numpy as np
from PIL import Image
from typing import Union


def fft_highpass_residual(
    img: Union[str, Image.Image, np.ndarray]
) -> np.ndarray:
    """
    Apply FFT high-pass filter to extract high-frequency residuals.
    
    This method transforms the image to the frequency domain using Fast Fourier Transform (FFT),
    removes low-frequency components by zeroing out the center region, and transforms back to
    spatial domain. The resulting high-frequency residual emphasizes manipulation boundaries,
    texture anomalies, and artifacts that are difficult to detect in the spatial domain.
    
    Parameters
    ----------
    img : str, PIL.Image.Image, or np.ndarray
        Input image. Can be:
        - File path (str): Path to the image file
        - PIL Image: Color or grayscale PIL Image object
        - NumPy array: (H, W) or (H, W, C) array
        Image will be converted to grayscale internally.
    
    Returns
    -------
    residual : np.ndarray
        High-frequency residual as 2D grayscale array (H, W) of type float32.
        - Contains raw residual values (may be positive or negative)
        - Higher absolute values indicate stronger high-frequency components
        - Typical usage: Apply contrast enhancement for visualization
    
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
    >>> residual = fft_highpass_residual(img)
    >>> 
    >>> # Enhance contrast for visualization
    >>> mean_val = np.mean(residual)
    >>> std_val = np.std(residual)
    >>> residual_enhanced = np.clip((residual - mean_val) / (3*std_val) + 0.5, 0, 1)
    >>> 
    >>> plt.imshow(residual_enhanced, cmap='gray')
    >>> plt.title("FFT High-Pass Residual")
    >>> plt.show()
    
    >>> # Example 2: Using file path
    >>> residual = fft_highpass_residual("image.jpg")
    
    >>> # Example 3: Display without enhancement
    >>> residual = fft_highpass_residual(img)
    >>> plt.imshow(np.abs(residual), cmap='hot')
    >>> plt.title("FFT High-Pass Residual (Absolute)")
    >>> plt.colorbar()
    >>> plt.show()
    
    >>> # Example 4: Using NumPy array
    >>> img_array = np.random.rand(512, 512, 3) * 255
    >>> residual = fft_highpass_residual(img_array)
    
    >>> # Example 5: Compare with original
    >>> fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    >>> axes[0].imshow(img, cmap='gray')
    >>> axes[0].set_title('Original Image')
    >>> axes[1].imshow(residual_enhanced, cmap='gray')
    >>> axes[1].set_title('FFT High-Pass Residual')
    >>> plt.show()
    
    Notes
    -----
    - FFT high-pass filtering is effective for detecting:
      * Deepfake and face forgery artifacts
      * Splicing and copy-paste boundaries
      * Synthetic image patterns (GAN, diffusion models)
      * Texture inconsistencies and resampling artifacts
      * Manipulation traces independent of JPEG history
    
    - Technical details:
      * Uses center_keep_ratio=0.25 (removes central 25% of frequency spectrum)
      * Low frequencies (DC component) are zeroed out in the center
      * High frequencies at edges are preserved
      * IFFT reconstructs spatial domain with only high-frequency components
    
    - The input image is converted to grayscale using standard weights:
      Gray = 0.299*R + 0.587*G + 0.114*B
    
    - Raw residuals are NOT normalized - contrast enhancement is recommended:
```python
      mean_val = np.mean(residual)
      std_val = np.std(residual)
      residual_enhanced = np.clip((residual - mean_val) / (3*std_val) + 0.5, 0, 1)
```
    
    - This method is JPEG-agnostic: works on both JPEG and uncompressed images
    
    - Frequency-domain analysis has been proven effective in:
      * Face forgery detection (CVPR 2020, 2021)
      * Deepfake generalization (AAAI 2024)
      * General image manipulation detection (ECCV 2020)
    
    See Also
    --------
    numpy.fft.fft2 : 2D Fast Fourier Transform
    numpy.fft.fftshift : Shift zero-frequency component to center
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
        arr = np.array(img.convert("L"), dtype=np.float32)
    else:
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 3:
            # Convert RGB to grayscale
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    
    # Get image dimensions
    H, W = arr.shape
    
    # Apply FFT and shift zero frequency to center
    F = np.fft.fftshift(np.fft.fft2(arr))
    
    # Remove low frequencies (center region)
    center_keep_ratio = 0.25
    hk = int(H * center_keep_ratio / 2.0)
    wk = int(W * center_keep_ratio / 2.0)
    F[H//2 - hk:H//2 + hk, W//2 - wk:W//2 + wk] = 0
    
    # Inverse FFT to get high-frequency residual
    residual = np.fft.ifft2(np.fft.ifftshift(F)).real.astype(np.float32)
    
    return residual


# Function schema for agent integration
FFT_HIGHPASS_RESIDUAL_SCHEMA = {
    "name": "fft_highpass_residual",
    "description": "Apply FFT high-pass filter to extract high-frequency residuals, revealing forgery boundaries, deepfake artifacts, and texture anomalies. Works independently of JPEG compression history.",
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
        "description": "2D high-frequency residual map (H×W, float32) with raw values. "
    }
}


if __name__ == "__main__":
    """
    Example usage and visualization
    """
    import matplotlib.pyplot as plt
    
    # Example: Apply FFT high-pass filter and visualize
    img_path = "temp_ela.jpg"  # Replace with your image path
    
    try:
        # Load image
        img = Image.open(img_path)
        
        # Apply FFT high-pass filtering
        residual = fft_highpass_residual(img)
        
        # Enhance contrast for visualization
        mean_val = np.mean(residual)
        std_val = np.std(residual)
        residual_enhanced = np.clip((residual - mean_val) / (3 * std_val) + 0.5, 0, 1)
        
        # Visualize original and residual side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Original image
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # FFT high-pass residual
        axes[1].imshow(residual_enhanced, cmap='gray')
        axes[1].set_title('FFT High-Pass Residual (Enhanced)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig("fft_highpass_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ FFT high-pass residual generated successfully!")
        print(f"✓ Comparison saved to fft_highpass_comparison.png")
        print(f"✓ Residual statistics: mean={mean_val:.2f}, std={std_val:.2f}")
        
        # Display residual alone with colorbar
        plt.figure(figsize=(10, 8))
        plt.imshow(residual_enhanced, cmap='gray')
        plt.title("FFT High-Pass Residual")
        plt.colorbar(label='Normalized Intensity')
        plt.axis('off')
        plt.savefig("fft_highpass_residual.png", dpi=150, bbox_inches='tight')
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found.")
        print("Please update 'img_path' in the __main__ section with a valid image path.")
    except Exception as e:
        print(f"Error: {e}")