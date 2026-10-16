"""
JPEG Ghost Map for Image Forensics

This module implements JPEG Ghost detection (also known as JPEG re-compression error map),
which is part of the ELA (Error Level Analysis) family. It reveals regions with different
compression histories, useful for detecting splicing and composite forgeries.

Principle:
    Re-compress the image at different JPEG quality levels and compute the absolute 
    difference between the re-compressed version and the original. Regions that have 
    been previously compressed at different quality levels will show different error patterns.

"""

import io
import numpy as np
from PIL import Image
from typing import Union, List, Dict, Literal


def jpeg_ghost_map(
    img: Union[str, Image.Image, np.ndarray],
    qualities: List[Literal[75, 85, 95]]
) -> Dict[int, np.ndarray]:
    """
    Generate JPEG ghost maps by re-compressing the image at different quality levels.
    
    JPEG ghost detection works by re-compressing an image at various JPEG quality levels
    and computing the absolute difference from the original. Authentic regions and tampered
    regions will exhibit different error patterns because they have different compression
    histories.
    
    Parameters
    ----------
    img : str, PIL.Image.Image, or np.ndarray
        Input image. Can be:
        - File path (str): Path to the image file
        - PIL Image: RGB or other PIL Image object
        - NumPy array: (H, W, C) array
        
    qualities : list of {75, 85, 95}
        List of JPEG quality levels for re-compression. Must contain values from {75, 85, 95}.
        - 75: Lower quality, more compression artifacts, higher sensitivity
        - 85: Medium quality, balanced detection (commonly used)
        - 95: Higher quality, subtle differences, less noise
        Typical usage: [75, 85, 95] to compare across multiple quality levels
    
    Returns
    -------
    ghost_maps : dict[int, np.ndarray]
        Dictionary mapping quality level to ghost map.
        - Keys: quality levels (e.g., 75, 85, 95)
        - Values: 2D numpy arrays (H, W) of type float32
        - Each ghost map contains the mean absolute error across RGB channels
        - Higher values indicate larger differences (potential tampering)
    
    Raises
    ------
    ValueError
        If any quality value is not in {75, 85, 95}
    FileNotFoundError
        If img is a file path that doesn't exist
    
    Examples
    --------
    >>> import numpy as np
    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    
    >>> # Example 1: Basic usage with file path
    >>> ghost_maps = jpeg_ghost_map("suspicious_image.jpg", qualities=[75, 85, 95])
    >>> plt.imshow(ghost_maps[85], cmap='hot')
    >>> plt.title("JPEG Ghost Map (Quality=85)")
    >>> plt.show()
    
    >>> # Example 2: Using PIL Image
    >>> img = Image.open("image.jpg")
    >>> ghost_maps = jpeg_ghost_map(img, qualities=[75, 85, 95])
    
    >>> # Example 3: Compare different quality levels
    >>> fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    >>> for idx, q in enumerate([75, 85, 95]):
    ...     axes[idx].imshow(ghost_maps[q], cmap='hot')
    ...     axes[idx].set_title(f'JPEG Ghost Map (Quality {q})')
    ...     axes[idx].axis('off')
    >>> plt.show()
    
    >>> # Example 4: Single quality level
    >>> ghost_map = jpeg_ghost_map(img, qualities=[85])
    >>> plt.imshow(ghost_map[85], cmap='hot')
    >>> plt.show()
    
    >>> # Example 5: Using NumPy array
    >>> img_array = np.random.rand(512, 512, 3) * 255
    >>> ghost_maps = jpeg_ghost_map(img_array, qualities=[75, 85])
    
    Notes
    -----
    - JPEG ghost detection is effective for:
      * Detecting spliced regions from different source images
      * Identifying areas that have been compressed at different quality levels
      * Revealing copy-paste forgeries with inconsistent compression history
    
    - Interpretation:
      * Uniform low error: Likely authentic or consistently compressed
      * High error in specific regions: Potential tampering or splicing
      * Different error patterns across qualities: Different compression histories
    
    - The ghost map shows the mean absolute difference across RGB channels
    - Each quality level produces a separate ghost map for comparison
    - Lower quality values (e.g., 75) produce more visible artifacts
    - Higher quality values (e.g., 95) show more subtle differences
    
    - The function uses JPEG subsampling=0 (no chroma subsampling) for more accurate results
    
    See Also
    --------
    PIL.Image.save : Used for JPEG compression
    
    """
    
    # Convert to PIL Image if necessary
    im = img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(np.uint8(img)).convert("RGB")
    
    # Get base image as float array
    base = np.array(im, np.float32)
    
    # Generate ghost maps for each quality level
    out = {}
    for q in qualities:
        # Re-compress at quality q
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q, subsampling=0)
        
        # Reload the re-compressed image
        rec = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        
        # Compute mean absolute error across RGB channels
        out[q] = np.abs(np.array(rec, np.float32) - base).mean(-1).astype(np.float32)  # (H, W)
    
    return out


# Function schema for agent integration
JPEG_GHOST_MAP_SCHEMA = {
    "name": "jpeg_ghost_map",
    "description": "Generate JPEG ghost maps by re-compressing image at different quality levels. Reveals regions with different compression histories, useful for detecting splicing and composite forgeries.",
    "parameters": {
        "type": "object",
        "properties": {
            "img": {
                "type": ["string", "object"],
                "description": "Input image. Can be file path (string), PIL Image object, or NumPy array"
            },
            "qualities": {
                "type": "array",
                "items": {
                    "type": "integer",
                    "enum": [75, 85, 95]
                },
                "description": "List of JPEG quality levels for re-compression. 75=high sensitivity, 85=balanced (recommended), 95=subtle differences. Typical: [75, 85, 95]"
            }
        },
        "required": ["img", "qualities"]
    },
    "returns": {
        "type": "object",
        "description": "Dictionary mapping quality level (int) to ghost map (2D NumPy array, float32). Higher values indicate larger differences (potential tampering)."
    }
}


if __name__ == "__main__":
    """
    Example usage and visualization
    """
    import matplotlib.pyplot as plt
    
    # Example: Generate and visualize JPEG ghost maps
    img_path = "temp_ela.jpg"  # Replace with your image path
    
    try:
        img = Image.open(img_path)
        
        # Calculate JPEG Ghost Maps with default quality parameters
        ghost_maps = jpeg_ghost_map(img, qualities=[75, 85, 95])
        
        # Visualize the ghost maps
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for idx, q in enumerate([75, 85, 95]):
            axes[idx].imshow(ghost_maps[q], cmap='hot')
            axes[idx].set_title(f'JPEG Ghost Map (Quality {q})')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig("jpeg_ghost_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ JPEG ghost maps generated successfully!")
        print(f"✓ Comparison saved to jpeg_ghost_comparison.png")
        print(f"✓ Available quality levels: {list(ghost_maps.keys())}")
        
        # Display individual ghost map (most commonly used: quality=85)
        plt.figure(figsize=(10, 8))
        plt.imshow(ghost_maps[85], cmap='hot')
        plt.title("JPEG Ghost Map (Quality=85)")
        plt.colorbar(label='Mean Absolute Error')
        plt.axis('off')
        plt.savefig("jpeg_ghost_q85.png", dpi=150, bbox_inches='tight')
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found.")
        print("Please update 'img_path' in the __main__ section with a valid image path.")
    except Exception as e:
        print(f"Error: {e}")