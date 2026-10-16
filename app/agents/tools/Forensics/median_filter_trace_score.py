"""
Median Filter Trace Detection for Image Forensics

This module implements median filter trace detection based on first-order difference
histogram analysis. Median filtering is commonly used in image manipulation to smooth
artifacts, remove noise, or blend spliced regions. This detection method exploits the
characteristic changes median filtering introduces to the pixel difference distribution.

Principle:
    Compute first-order differences (gradients) in horizontal and vertical directions.
    Analyze the histogram of these differences, focusing on the probabilities of zero
    differences (p0) and ±1 differences (p±1). Median filtering increases the proportion
    of zero differences relative to ±1 differences, creating a distinctive signature.

"""

import numpy as np
from PIL import Image
from typing import Union


def median_filter_trace_score(
    img: Union[str, Image.Image, np.ndarray]
) -> float:
    """
    Detect median filtering traces by analyzing first-order difference histogram.
    
    Median filtering is a common post-processing operation in image manipulation used to
    smooth artifacts, remove noise, or blend spliced regions. This function detects median
    filtering by examining the distribution of pixel differences. Median filtering creates
    a characteristic pattern: it increases zero differences (p0) while decreasing ±1
    differences (p±1), resulting in a higher p0/p±1 ratio.
    
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
    score : float
        Median filter trace score (p0/p±1 ratio).
        - Values typically range from 0.5 to 5.0+
        - score > 1.5: Likely median filtered (suspicious)
        - score ≈ 1.0: Natural image (normal)
        - score < 0.8: Unusual, possibly other processing
        - Higher scores indicate stronger median filtering traces
    
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
    >>> from scipy.ndimage import median_filter
    
    >>> # Example 1: Test on natural image
    >>> img = Image.open("natural_image.jpg")
    >>> score = median_filter_trace_score(img)
    >>> print(f"Median filter score: {score:.4f}")
    >>> if score > 1.5:
    ...     print("Warning: Possible median filtering detected!")
    >>> else:
    ...     print("Image appears natural.")
    
    >>> # Example 2: Compare original vs median filtered
    >>> img_original = Image.open("image.jpg")
    >>> score_original = median_filter_trace_score(img_original)
    >>> 
    >>> # Apply median filter
    >>> img_array = np.array(img_original)
    >>> img_filtered = median_filter(img_array, size=3)
    >>> score_filtered = median_filter_trace_score(img_filtered)
    >>> 
    >>> print(f"Original score: {score_original:.4f}")
    >>> print(f"Filtered score: {score_filtered:.4f}")
    >>> print(f"Increase: {score_filtered - score_original:.4f}")
    
    >>> # Example 3: Using file path
    >>> score = median_filter_trace_score("suspicious_image.jpg")
    
    >>> # Example 4: Batch processing multiple images
    >>> import os
    >>> for filename in os.listdir("images/"):
    ...     if filename.endswith(".jpg"):
    ...         score = median_filter_trace_score(f"images/{filename}")
    ...         if score > 1.5:
    ...             print(f"{filename}: SUSPICIOUS (score={score:.2f})")
    
    >>> # Example 5: Using NumPy array
    >>> img_array = np.random.rand(512, 512, 3) * 255
    >>> score = median_filter_trace_score(img_array)
    
    Notes
    -----
    - Median filter traces are effective for detecting:
      * Post-processing to smooth manipulation artifacts
      * Blending operations in spliced regions
      * Noise removal after tampering
      * Image enhancement operations
      * Forensic anti-forensics attempts
    
    - Technical details:
      * Computes horizontal differences: dx = X[i,j+1] - X[i,j]
      * Computes vertical differences: dy = X[i+1,j] - X[i,j]
      * Builds histogram of differences rounded to integers
      * Focuses on differences in range [-5, 5]
      * Calculates p0 = P(difference = 0)
      * Calculates p±1 = P(difference = ±1)
      * Score = p0 / p±1
    
    - Why median filtering is detectable:
      * Median filtering replaces pixels with neighborhood median
      * Creates more flat regions (zero differences)
      * Reduces small gradients (±1 differences)
      * Changes the natural statistics of pixel differences
      * Leaves a characteristic "footprint" in difference histogram
    
    - Interpretation guidelines:
      * score > 2.0: Strong evidence of median filtering
      * 1.5 < score ≤ 2.0: Moderate evidence, likely filtered
      * 1.0 < score ≤ 1.5: Weak evidence, borderline
      * 0.8 ≤ score ≤ 1.0: Normal range for natural images
      * score < 0.8: Unusual, possibly sharpened or other processing
    
    - Limitations:
      * May produce false positives on:
        - Very smooth images (sky, walls, gradients)
        - Images with large uniform regions
        - Heavy JPEG compression (creates similar effects)
      * May miss detection on:
        - Very small median filter kernels (1×1, 2×2)
        - Images with subsequent processing
        - Spatially localized filtering (only affects average score)
    
    - The input image is converted to grayscale using standard weights:
      Gray = 0.299*R + 0.587*G + 0.114*B
    
    - This method is based on the observation that median filtering creates
      a characteristic "peak" at zero in the difference histogram
    
    See Also
    --------
    numpy.diff : Compute differences
    scipy.ndimage.median_filter : Apply median filtering
    """
    # Convert to grayscale numpy array
    if isinstance(img, Image.Image):
        X = np.array(img.convert("L"), np.float32)
    else:
        X = np.asarray(img, np.float32)
        # Convert RGB to grayscale if needed
        if X.ndim == 3:
            X = 0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2]
    
    # Compute first-order differences
    # Horizontal differences: X[i, j+1] - X[i, j]
    dx = np.diff(X, axis=1)
    # Vertical differences: X[i+1, j] - X[i, j]
    dy = np.diff(X, axis=0)
    
    # Combine all differences
    d = np.concatenate([dx.ravel(), dy.ravel()])
    
    # Round to nearest integer and clip to [-5, 5] range
    d_rounded = np.clip(np.round(d).astype(int), -5, 5)
    
    # Build histogram (11 bins for values from -5 to 5)
    counts = np.bincount(d_rounded + 5, minlength=11).astype(np.float32)
    
    # Calculate probabilities
    total = counts.sum() + 1e-8
    
    # p0: probability of zero difference (index 5 corresponds to difference = 0)
    p0 = counts[5] / total
    
    # p±1: probability of ±1 differences (indices 4 and 6 correspond to -1 and +1)
    p_pm1 = (counts[4] + counts[6]) / total
    
    # Compute median filter trace score (p0 / p±1 ratio)
    score = float(p0 / (p_pm1 + 1e-8))
    
    return score


# Function schema for agent integration
MEDIAN_FILTER_TRACE_SCORE_SCHEMA = {
    "name": "median_filter_trace_score",
    "description": "Detect median filtering traces by analyzing first-order difference histogram. Returns score where higher values indicate stronger evidence of median filtering. score > 1.5 suggests likely filtering; score ≈ 1.0 indicates natural image.",
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
        "description": "Median filter trace score (float). score > 1.5 = likely filtered; score ≈ 1.0 = natural; higher scores indicate stronger median filtering traces."
    }
}


if __name__ == "__main__":
    """
    Example usage and demonstration
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import median_filter as scipy_median_filter
    
    # Example: Detect median filtering traces
    img_path = "temp_ela.jpg"  # Replace with your image path
    
    try:
        # Load image
        img = Image.open(img_path)
        
        # Test on original image
        score_original = median_filter_trace_score(img)
        
        print("=" * 60)
        print("Median Filter Trace Detection Results")
        print("=" * 60)
        print(f"\nOriginal image score: {score_original:.4f}")
        
        # Interpretation
        if score_original > 2.0:
            print("  ⚠️  Strong evidence of median filtering")
        elif score_original > 1.5:
            print("  ⚠️  Moderate evidence, likely filtered")
        elif score_original > 1.0:
            print("  ⚡ Weak evidence, borderline case")
        else:
            print("  ✓  Normal range, likely natural image")
        
        # Create median filtered versions for comparison
        img_array = np.array(img.convert("L"))
        img_filtered_3x3 = scipy_median_filter(img_array, size=3)
        img_filtered_5x5 = scipy_median_filter(img_array, size=5)
        
        score_filtered_3x3 = median_filter_trace_score(img_filtered_3x3)
        score_filtered_5x5 = median_filter_trace_score(img_filtered_5x5)
        
        print(f"\nAfter 3×3 median filter: {score_filtered_3x3:.4f} (increase: {score_filtered_3x3 - score_original:+.4f})")
        print(f"After 5×5 median filter: {score_filtered_5x5:.4f} (increase: {score_filtered_5x5 - score_original:+.4f})")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title(f'Original (score={score_original:.3f})')
        axes[0].axis('off')
        
        axes[1].imshow(img_filtered_3x3, cmap='gray')
        axes[1].set_title(f'Median 3×3 (score={score_filtered_3x3:.3f})')
        axes[1].axis('off')
        
        axes[2].imshow(img_filtered_5x5, cmap='gray')
        axes[2].set_title(f'Median 5×5 (score={score_filtered_5x5:.3f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig("median_filter_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Comparison saved to median_filter_comparison.png")
        
        # Print interpretation guide
        print("\n" + "=" * 60)
        print("Score Interpretation Guide:")
        print("=" * 60)
        print("  score > 2.0       : Strong evidence of median filtering")
        print("  1.5 < score ≤ 2.0 : Moderate evidence, likely filtered")
        print("  1.0 < score ≤ 1.5 : Weak evidence, borderline case")
        print("  0.8 ≤ score ≤ 1.0 : Normal range for natural images")
        print("  score < 0.8       : Unusual, possibly other processing")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found.")
        print("Please update 'img_path' in the __main__ section with a valid image path.")
    except Exception as e:
        print(f"Error: {e}")