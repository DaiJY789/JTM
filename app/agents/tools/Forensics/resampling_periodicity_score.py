"""
Resampling Periodicity Detection for Image Forensics

This module implements resampling trace detection based on periodic patterns in prediction
errors. Geometric transformations (scaling, rotation) that use interpolation introduce
periodic artifacts in the image's prediction error field. This method detects these
periodicities by analyzing the frequency spectrum of Radon projections at multiple angles.

Principle:
    Apply Laplacian filter to extract prediction errors, then compute Radon projections
    (line integrals) at various angles. Analyze the frequency spectrum of each projection
    to detect periodic peaks. Resampled images exhibit stronger periodic components due to
    interpolation-induced correlations between neighboring pixels.

"""

import numpy as np
from PIL import Image
from typing import Union


def resampling_periodicity_score(
    img: Union[str, Image.Image, np.ndarray]
) -> float:
    """
    Detect resampling traces by analyzing periodic patterns in prediction errors.
    
    Geometric transformations (scaling, rotation) that use interpolation introduce periodic
    artifacts in the prediction error field. This function detects resampling by computing
    Radon projections at multiple angles and analyzing their frequency spectra for periodic
    peaks. Higher peak values indicate stronger evidence of resampling.
    
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
    peak : float
        Resampling periodicity score (maximum spectral peak across all angles).
        - Values typically range from 1.0 to 10.0+
        - peak > 3.0: Likely resampled (suspicious)
        - peak ≈ 1.0-2.0: Natural image (normal)
        - peak < 1.0: Unusual, possibly heavily compressed
        - Higher peaks indicate stronger resampling traces
    
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
    >>> from scipy.ndimage import zoom
    
    >>> # Example 1: Test on natural image
    >>> img = Image.open("natural_image.jpg")
    >>> score = resampling_periodicity_score(img)
    >>> print(f"Resampling score: {score:.4f}")
    >>> if score > 3.0:
    ...     print("Warning: Possible resampling detected!")
    >>> else:
    ...     print("Image appears natural.")
    
    >>> # Example 2: Compare original vs resampled
    >>> img_original = Image.open("image.jpg")
    >>> score_original = resampling_periodicity_score(img_original)
    >>> 
    >>> # Apply resampling (scale down then up)
    >>> img_array = np.array(img_original)
    >>> img_resampled = zoom(zoom(img_array, 0.75, order=1), 1.33, order=1)
    >>> score_resampled = resampling_periodicity_score(img_resampled)
    >>> 
    >>> print(f"Original score: {score_original:.4f}")
    >>> print(f"Resampled score: {score_resampled:.4f}")
    >>> print(f"Increase: {score_resampled - score_original:.4f}")
    
    >>> # Example 3: Using file path
    >>> score = resampling_periodicity_score("suspicious_image.jpg")
    
    >>> # Example 4: Test rotated image
    >>> img_rotated = img_original.rotate(15, resample=Image.BICUBIC)
    >>> score_rotated = resampling_periodicity_score(img_rotated)
    >>> print(f"Rotated image score: {score_rotated:.4f}")
    
    >>> # Example 5: Batch processing
    >>> import os
    >>> for filename in os.listdir("images/"):
    ...     if filename.endswith(".jpg"):
    ...         score = resampling_periodicity_score(f"images/{filename}")
    ...         if score > 3.0:
    ...             print(f"{filename}: SUSPICIOUS (score={score:.2f})")
    
    Notes
    -----
    - Resampling periodicity is effective for detecting:
      * Image scaling (upsampling and downsampling)
      * Image rotation with interpolation
      * Geometric transformations (affine, perspective)
      * Spliced regions from different source resolutions
      * Content-aware scaling artifacts
    
    - Technical details:
      * Applies 2D Laplacian filter to extract prediction errors
      * Computes projections at angles: 0°, 15°, 30°, ..., 165° (12 angles)
      * Analyzes frequency spectrum of each projection
      * Detects periodic peaks caused by interpolation
      * Returns maximum peak value across all angles
    
    - Why resampling is detectable:
      * Interpolation creates correlations between neighboring pixels
      * These correlations form periodic patterns in prediction errors
      * Periodicity direction depends on resampling orientation
      * Natural images lack such regular periodic structures
    
    - Laplacian prediction error kernel:
```
      [ 0  1  0 ]
      [ 1 -4  1 ]
      [ 0  1  0 ]
```
      This approximates prediction error: pixel - neighborhood average
    
    - Interpretation guidelines:
      * peak > 5.0: Strong evidence of resampling
      * 3.0 < peak ≤ 5.0: Moderate evidence, likely resampled
      * 2.0 < peak ≤ 3.0: Weak evidence, borderline
      * 1.0 ≤ peak ≤ 2.0: Normal range for natural images
      * peak < 1.0: Unusual, possibly heavily processed/compressed
    
    - Limitations:
      * May produce false positives on:
        - Images with strong regular patterns (textures, grids)
        - Screen captures with aliasing
        - Heavy JPEG compression (disrupts periodicity)
      * May miss detection on:
        - Very small scaling factors (close to 1.0)
        - Multiple cascaded operations
        - Anti-forensic techniques specifically designed to remove traces
    
    - The input image is converted to grayscale using standard weights:
      Gray = 0.299*R + 0.587*G + 0.114*B
    
    - Analysis is performed at 12 angles (0° to 165° in 15° steps) to capture
      periodicity in any direction
    
    See Also
    --------
    numpy.fft.rfft : Real FFT for frequency analysis
    PIL.Image.rotate : Image rotation
    """ 
    # Convert to grayscale numpy array
    if isinstance(img, Image.Image):
        X = np.array(img.convert("L"), np.float32)
    else:
        X = np.asarray(img, np.float32)
        # Convert RGB to grayscale if needed
        if X.ndim == 3:
            X = 0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2]
    
    # Define Laplacian kernel for prediction error
    K = np.array([[0, 1, 0], 
                  [1, -4, 1], 
                  [0, 1, 0]], np.float32)
    
    # 2D convolution function with reflection padding
    def conv2(x, k):
        H, W = x.shape
        kh, kw = k.shape
        ph, pw = kh // 2, kw // 2
        xp = np.pad(x, ((ph, ph), (pw, pw)), 'reflect')
        y = np.zeros_like(x)
        for i in range(H):
            for j in range(W):
                y[i, j] = np.sum(xp[i:i+kh, j:j+kw] * k)
        return y
    
    # Compute prediction error field
    E = conv2(X, K)
    
    # Test multiple angles (0° to 165° in 15° steps)
    angles = range(0, 180, 15)
    max_peak = 0.0
    
    for angle in angles:
        # Rotate error field
        E_rotated = np.array(
            Image.fromarray(E).rotate(angle, resample=Image.NEAREST, expand=True), 
            np.float32
        )
        
        # Compute projection (sum along columns)
        projection = E_rotated.sum(axis=0)
        
        # Analyze frequency spectrum
        projection_centered = projection - projection.mean()
        spectrum = np.abs(np.fft.rfft(projection_centered))
        
        # Ignore DC component
        spectrum[0] = 0
        
        # Compute peak-to-mean ratio
        peak = float(spectrum.max() / (spectrum.mean() + 1e-8))
        
        # Track maximum peak across all angles
        if peak > max_peak:
            max_peak = peak
    
    return max_peak


# Function schema for agent integration
RESAMPLING_PERIODICITY_SCORE_SCHEMA = {
    "name": "resampling_periodicity_score",
    "description": "Detect resampling traces (scaling, rotation) by analyzing periodic patterns in prediction errors. Returns score where higher values indicate stronger evidence of resampling. peak > 3.0 suggests likely resampling; peak ≈ 1.0-2.0 indicates natural image.",
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
        "type": "number",
        "description": "Resampling periodicity score (float). peak > 3.0 = likely resampled; peak ≈ 1.0-2.0 = natural; higher peaks indicate stronger resampling traces."
    }
}


if __name__ == "__main__":
    """
    Example usage and demonstration
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    
    # Example: Detect resampling traces
    img_path = "temp_ela.jpg"  # Replace with your image path
    
    try:
        # Load image
        img = Image.open(img_path)
        
        # Test on original image
        score_original = resampling_periodicity_score(img)
        
        print("=" * 60)
        print("Resampling Periodicity Detection Results")
        print("=" * 60)
        print(f"\nOriginal image score: {score_original:.4f}")
        
        # Interpretation
        if score_original > 5.0:
            print("  ⚠️  Strong evidence of resampling")
        elif score_original > 3.0:
            print("  ⚠️  Moderate evidence, likely resampled")
        elif score_original > 2.0:
            print("  ⚡ Weak evidence, borderline case")
        else:
            print("  ✓  Normal range, likely natural image")
        
        # Create resampled versions for comparison
        img_array = np.array(img.convert("L"))
        
        # Downscale then upscale (simulates resampling)
        img_resampled_75 = zoom(zoom(img_array, 0.75, order=1), 1.333, order=1)
        img_resampled_75 = img_resampled_75[:img_array.shape[0], :img_array.shape[1]]
        
        # Rotate (also causes resampling)
        img_rotated = img.rotate(15, resample=Image.BICUBIC)
        
        score_resampled = resampling_periodicity_score(img_resampled_75)
        score_rotated = resampling_periodicity_score(img_rotated)
        
        print(f"\nAfter 75% downscale + upscale: {score_resampled:.4f} (change: {score_resampled - score_original:+.4f})")
        print(f"After 15° rotation: {score_rotated:.4f} (change: {score_rotated - score_original:+.4f})")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title(f'Original (score={score_original:.3f})')
        axes[0].axis('off')
        
        axes[1].imshow(img_resampled_75, cmap='gray')
        axes[1].set_title(f'Resampled 75% (score={score_resampled:.3f})')
        axes[1].axis('off')
        
        axes[2].imshow(img_rotated, cmap='gray')
        axes[2].set_title(f'Rotated 15° (score={score_rotated:.3f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig("resampling_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Comparison saved to resampling_comparison.png")
        
        # Print interpretation guide
        print("\n" + "=" * 60)
        print("Score Interpretation Guide:")
        print("=" * 60)
        print("  peak > 5.0        : Strong evidence of resampling")
        print("  3.0 < peak ≤ 5.0  : Moderate evidence, likely resampled")
        print("  2.0 < peak ≤ 3.0  : Weak evidence, borderline case")
        print("  1.0 ≤ peak ≤ 2.0  : Normal range for natural images")
        print("  peak < 1.0        : Unusual, possibly heavily processed")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found.")
        print("Please update 'img_path' in the __main__ section with a valid image path.")
    except Exception as e:
        print(f"Error: {e}")