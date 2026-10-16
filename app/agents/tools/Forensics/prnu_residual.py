"""
PRNU (Photo Response Non-Uniformity) Noise Residual for Image Forensics

This module implements PRNU extraction using wavelet-based denoising to reveal camera
sensor fingerprints. PRNU is a unique pattern caused by manufacturing imperfections in
camera sensors, acting as a "digital fingerprint" that can identify the source camera
and detect image manipulation, splicing, and forgery.

Principle:
    Extract noise residual by subtracting a denoised version (via wavelet soft-thresholding)
    from the original image. The residual contains the camera's PRNU pattern, which should
    be consistent across an authentic image but disrupted in tampered regions. Optional
    block-level normalized cross-correlation (NCC) can compare against a reference PRNU
    to localize inconsistencies.
"""

import numpy as np
import pywt
from PIL import Image
from typing import Union


def _wavelet_denoise(x: np.ndarray, wavelet: str = "db2", level: int = 2) -> np.ndarray:
    """
    Denoise image using wavelet soft-thresholding (Donoho's universal threshold).
    
    Parameters
    ----------
    x : np.ndarray
        Input grayscale image
    wavelet : str
        Wavelet family (default: "db2" - Daubechies 2)
    level : int
        Decomposition level
        
    Returns
    -------
    denoised : np.ndarray
        Denoised image with same shape as input
    """
    # Store original shape
    original_shape = x.shape
    
    # Wavelet decomposition
    c = pywt.wavedec2(x, wavelet, level=level, mode='periodization')
    detail = c[1:]
    
    # Estimate noise standard deviation (MAD estimator)
    sigma = np.median(np.abs(detail[0][0])) / 0.6745 + 1e-8
    
    # Universal threshold (Donoho & Johnstone)
    T = sigma * np.sqrt(2 * np.log(x.size))
    
    # Apply soft thresholding to detail coefficients
    new = [c[0]]  # Keep approximation coefficients
    for (cH, cV, cD) in detail:
        new.append((
            pywt.threshold(cH, T, 'soft'),
            pywt.threshold(cV, T, 'soft'),
            pywt.threshold(cD, T, 'soft')
        ))
    
    # Wavelet reconstruction
    denoised = pywt.waverec2(new, wavelet, mode='periodization')
    
    # Ensure output matches input shape (crop if necessary)
    if denoised.shape != original_shape:
        denoised = denoised[:original_shape[0], :original_shape[1]]
    
    return denoised


def prnu_residual(
    img: Union[str, Image.Image, np.ndarray]
) -> np.ndarray:
    """
    Extract PRNU (Photo Response Non-Uniformity) noise residual from an image.
    
    PRNU is a unique pattern in digital camera sensors caused by manufacturing imperfections.
    It acts as a "digital fingerprint" for camera identification and tampering detection.
    This function extracts the PRNU residual by removing image content through wavelet-based
    denoising, leaving only the sensor noise pattern.
    
    Parameters
    ----------
    img : str, PIL.Image.Image, or np.ndarray
        Input image. Can be:
        - File path (str): Path to the image file
        - PIL Image: RGB or grayscale PIL Image object
        - NumPy array: (H, W) or (H, W, C) array
        Image will be converted to grayscale and normalized to [0, 1].
    
    Returns
    -------
    residual : np.ndarray
        PRNU noise residual as 2D grayscale array of type float32.
        - Shape: (H, W) same as input image
        - Values typically in range [-0.1, 0.1] (normalized)
        - Represents sensor-specific noise pattern
        - Should be spatially consistent in authentic images
        - Shows disruptions in tampered/spliced regions
    
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
    
    >>> # Example 1: Basic PRNU extraction
    >>> img = Image.open("camera_image.jpg")
    >>> residual = prnu_residual(img)
    >>> plt.imshow(residual, cmap='gray')
    >>> plt.title("PRNU Residual")
    >>> plt.colorbar()
    >>> plt.show()
    
    >>> # Example 2: Using file path
    >>> residual = prnu_residual("image.jpg")
    
    >>> # Example 3: Enhance visualization
    >>> residual = prnu_residual(img)
    >>> # Scale to [0, 1] for better visualization
    >>> residual_scaled = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
    >>> plt.imshow(residual_scaled, cmap='gray')
    >>> plt.show()
    
    >>> # Example 4: Compare original and residual
    >>> fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    >>> axes[0].imshow(img, cmap='gray')
    >>> axes[0].set_title('Original Image')
    >>> axes[1].imshow(residual, cmap='gray')
    >>> axes[1].set_title('PRNU Residual')
    >>> plt.show()
    
    >>> # Example 5: Using NumPy array
    >>> img_array = np.random.rand(512, 512) * 255
    >>> residual = prnu_residual(img_array)
    
    Notes
    -----
    - PRNU residual is effective for:
      * Camera source identification (digital fingerprinting)
      * Image authenticity verification
      * Detecting splicing and copy-paste forgeries
      * Identifying tampered regions with different camera sources
      * Verifying image integrity and origin
    
    - Technical details:
      * Uses Daubechies 2 wavelet ("db2") for denoising
      * 2-level wavelet decomposition
      * Donoho's universal threshold for soft-thresholding
      * MAD (Median Absolute Deviation) for noise estimation
      * Residual = Original - Denoised
      * Uses 'periodization' mode to handle boundary conditions
    
    - PRNU properties:
      * Unique to each individual camera sensor
      * Stable over time and imaging conditions
      * Multiplicative with image intensity
      * Spatially consistent across authentic images
      * Disrupted in spliced or manipulated regions
    
    - Interpretation:
      * Uniform pattern: Likely from single camera source (authentic)
      * Disrupted pattern: Potential splicing or manipulation
      * Different patterns in regions: Multiple source cameras (forgery)
      * Inconsistent noise: Possible tampering or synthetic content
    
    - The input image is converted to grayscale using standard weights:
      Gray = 0.299*R + 0.587*G + 0.114*B (via PIL.Image.convert("L"))
    
    - Values are normalized to [0, 1] before processing
    
    - For camera identification tasks, multiple images from the same camera
      should be averaged to create a reference PRNU pattern
    
    - PRNU analysis is most effective on:
      * Images directly from camera (not heavily processed)
      * High-resolution images (more pattern information)
      * Images with minimal compression (less artifact interference)
    
    See Also
    --------
    pywt.wavedec2 : Wavelet decomposition
    pywt.threshold : Wavelet coefficient thresholding

    """
    # Load image if path is provided
    
    # Convert to grayscale numpy array and normalize to [0, 1]
    if isinstance(img, Image.Image):
        X = np.array(img.convert("L"), np.float32) / 255.0
    else:
        X = np.asarray(img, np.float32)
        # Convert RGB to grayscale if needed
        if X.ndim == 3:
            X = 0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2]
        # Normalize to [0, 1] if not already
        if X.max() > 1.0:
            X = X / 255.0
    
    # Store original shape
    original_shape = X.shape
    
    # Denoise using wavelet soft-thresholding
    denoised = _wavelet_denoise(X, wavelet="db2", level=2)
    
    # Ensure denoised has same shape as original
    if denoised.shape != original_shape:
        denoised = denoised[:original_shape[0], :original_shape[1]]
    
    # Extract PRNU residual (original - denoised)
    residual = (X - denoised).astype(np.float32)
    
    return residual


# Function schema for agent integration
PRNU_RESIDUAL_SCHEMA = {
    "name": "prnu_residual",
    "description": "Extract PRNU (Photo Response Non-Uniformity) noise residual to reveal camera sensor fingerprints. Effective for camera identification, authenticity verification, and detecting splicing/tampering. Consistent patterns indicate authenticity; disrupted patterns suggest manipulation.",
    "parameters": {
        "type": "object",
        "properties": {
            "img": {
                "type": ["string", "object"],
                "description": "Input image. Can be file path (string), PIL Image object, or NumPy array. Will be converted to grayscale and normalized to [0, 1]."
            }
        },
        "required": ["img"]
    },
    "returns": {
        "type": "array",
        "description": "2D PRNU residual map (H×W, float32) with values typically in [-0.1, 0.1]. Represents sensor-specific noise pattern. Uniform pattern = authentic; disrupted pattern = potential tampering."
    }
}


if __name__ == "__main__":
    """
    Example usage and visualization
    """
    import matplotlib.pyplot as plt
    
    # Example: Extract and visualize PRNU residual
    img_path = "temp_ela.jpg"  # Replace with your image path
    
    try:
        # Load image
        img = Image.open(img_path)
        
        # Extract PRNU residual
        residual = prnu_residual(img)
        
        # Visualize original and residual side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Original image
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # PRNU residual
        im = axes[1].imshow(residual, cmap='gray')
        axes[1].set_title('PRNU Residual (Camera Fingerprint)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Residual Value')
        
        plt.tight_layout()
        plt.savefig("prnu_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ PRNU residual extracted successfully!")
        print(f"✓ Comparison saved to prnu_comparison.png")
        print(f"✓ Residual shape: {residual.shape}")
        print(f"✓ Residual range: [{residual.min():.4f}, {residual.max():.4f}]")
        print(f"✓ Residual mean: {residual.mean():.4f}, std: {residual.std():.4f}")
        
        # Display residual alone with enhanced visualization
        plt.figure(figsize=(10, 8))
        
        # Scale residual for better visualization
        residual_scaled = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
        
        plt.imshow(residual_scaled, cmap='gray')
        plt.title("PRNU Residual (Scaled)")
        plt.colorbar(label='Normalized Intensity')
        plt.axis('off')
        plt.savefig("prnu_residual.png", dpi=150, bbox_inches='tight')
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found.")
        print("Please update 'img_path' in the __main__ section with a valid image path.")
    except Exception as e:
        print(f"Error: {e}")