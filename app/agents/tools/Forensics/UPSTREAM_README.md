# Low-Level Image Forensics Toolkit

A comprehensive collection of 12 state-of-the-art image forensics tools for detecting manipulation, forgery, and tampering in digital images. Each tool is production-ready with standardized interfaces and includes function schemas for easy agent integration.

## 📋 Overview

This toolkit provides low-level forensic analysis methods covering frequency domain, spatial domain, noise pattern, and statistical approaches. All tools follow a unified design pattern for seamless integration into automated workflows.

## 🚀 Quick Start

### Installation
```bash
pip install numpy pillow scipy pywt matplotlib
```

### One-Command Execution

Each tool can be run independently:
```bash
# Example 1: Bayar Constrained Filter
python bayar_filter.py

# Example 2: JPEG Ghost Map
python jpeg_ghost_map.py

# Example 3: FFT High-Pass Filter
python fft_highpass_residual.py

# ... run any tool directly
```

All scripts include demo code in `if __name__ == "__main__":` blocks and will process `temp_ela.jpg` by default.

## 🛠️ Available Tools

### 1. **Bayar Constrained Filter** (`bayar_filter.py`)
- **Purpose**: Detect resampling, splicing, and statistical anomalies
- **Method**: Fixed prediction residual filter
- **Parameters**: `img` (required), `kernel_size` ∈ {3, 5, 7} (required)

### 2. **JPEG Ghost Map** (`jpeg_ghost_map.py`)
- **Purpose**: Reveal regions with different compression histories
- **Method**: Re-compression error analysis at multiple quality levels
- **Parameters**: `img` (required), `qualities` ∈ {75, 85, 95} (required)

### 3. **SRM Residual Bank** (`srm_residual_bank.py`)
- **Purpose**: Extract noise residuals for steganalysis and forensics
- **Method**: Spatial Rich Model with directional filters
- **Parameters**: `img` (required), `srm_filter` ∈ {0, 1, 2, 3} (required)

### 4. **FFT High-Pass Residual** (`fft_highpass_residual.py`)
- **Purpose**: Detect deepfakes, splicing, and synthetic patterns
- **Method**: Frequency domain high-pass filtering
- **Parameters**: `img` (required)

### 5. **DWT HH Subband** (`dwt_hh_map.py`)
- **Purpose**: Reveal texture inconsistencies and upsampling artifacts
- **Method**: Discrete Wavelet Transform high-frequency extraction
- **Parameters**: `img` (required), `level` ∈ {1, 2, 3} (required)

### 6. **Local Correlation Map** (`local_correlation_map.py`)
- **Purpose**: Detect AI-generated images and synthetic patterns
- **Method**: Local pixel correlation analysis
- **Parameters**: `img` (required), `w` ∈ {2, 3, 4, 5} (required)

### 7. **PRNU Residual** (`prnu_residual.py`)
- **Purpose**: Camera fingerprinting and source identification
- **Method**: Photo Response Non-Uniformity extraction
- **Parameters**: `img` (required)

### 8. **Median Filter Trace** (`median_filter_trace_score.py`)
- **Purpose**: Detect post-processing smoothing operations
- **Method**: First-order difference histogram analysis
- **Parameters**: `img` (required)

### 9. **Resampling Periodicity** (`resampling_periodicity_score.py`)
- **Purpose**: Detect scaling and rotation traces
- **Method**: Radon projection with spectral analysis
- **Parameters**: `img` (required)

### 10. **DCT High-Pass Filter** (`dct_highpass_filter.py`)
- **Purpose**: Reveal JPEG-related manipulation artifacts
- **Method**: DCT domain high-frequency extraction
- **Parameters**: `img` (required)

### 11. **Sobel Edge Detection** (`sobel_filter.py`)
- **Purpose**: Identify splicing boundaries and edge inconsistencies
- **Method**: Gradient magnitude computation
- **Parameters**: `img` (required)

### 12. **Laplacian High-Pass** (`highpass_filter.py`)
- **Purpose**: General-purpose high-frequency artifact detection
- **Method**: Second-order derivative filtering
- **Parameters**: `img` (required)

## 📦 Function Schema Integration

All tools provide standardized function schemas for agent/API integration:
```python
# Example: Bayar Filter Schema
from bayar_filter import BAYAR_FILTER_SCHEMA

print(BAYAR_FILTER_SCHEMA)
# {
#   "name": "bayar_filter",
#   "description": "Apply Bayar constrained filter...",
#   "parameters": {
#     "type": "object",
#     "properties": {
#       "img": {...},
#       "kernel_size": {"type": "integer", "enum": [3, 5, 7]}
#     },
#     "required": ["img", "kernel_size"]
#   },
#   "returns": {...}
# }
```

Available schemas:
- `BAYAR_FILTER_SCHEMA`
- `JPEG_GHOST_MAP_SCHEMA`
- `SRM_RESIDUAL_BANK_SCHEMA`
- `FFT_HIGHPASS_RESIDUAL_SCHEMA`
- `DWT_HH_MAP_SCHEMA`
- `LOCAL_CORRELATION_MAP_SCHEMA`
- `PRNU_RESIDUAL_SCHEMA`
- `MEDIAN_FILTER_TRACE_SCORE_SCHEMA`
- `RESAMPLING_PERIODICITY_SCORE_SCHEMA`
- `DCT_HIGHPASS_FILTER_SCHEMA`
- `SOBEL_FILTER_SCHEMA`
- `HIGHPASS_FILTER_SCHEMA`

## 💡 Usage Examples

### Basic Usage
```python
from bayar_filter import bayar_filter
from PIL import Image

# Load image
img = Image.open("suspicious_image.jpg")

# Apply filter
residual = bayar_filter(img, kernel_size=5)

# Visualize (normalized)
import matplotlib.pyplot as plt
residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
plt.imshow(residual_norm, cmap='gray')
plt.show()
```

### Pipeline Integration
```python
# Combine multiple tools for comprehensive analysis
from bayar_filter import bayar_filter
from prnu_residual import prnu_residual
from sobel_filter import sobel_filter

img = Image.open("test.jpg")

# Extract multiple forensic features
bayar_res = bayar_filter(img, kernel_size=5)
prnu_res = prnu_residual(img)
edges = sobel_filter(img)

```

## 📊 Output Formats

All tools return numpy arrays (float32) with raw values. 

## 🎯 Interpretation Guidelines

### Detection Thresholds (approximate):
- **Median Filter Trace**: score > 1.5 (likely filtered)
- **Resampling Periodicity**: peak > 3.0 (likely resampled)
- **Local Correlation**: higher values (suspicious for AI-generated)

### Visual Patterns:
- **Uniform patterns**: Likely authentic
- **Localized anomalies**: Potential tampering
- **Abrupt boundaries**: Splicing/copy-paste
- **Periodic structures**: Resampling artifacts

## 📁 File Structure
```
low_level_tool/
├── bayar_filter.py              # Tool 1
├── jpeg_ghost_map.py            # Tool 2
├── srm_residual_bank.py         # Tool 3
├── fft_highpass_residual.py     # Tool 4
├── dwt_hh_map.py                # Tool 5
├── local_correlation_map.py     # Tool 6
├── prnu_residual.py             # Tool 7
├── median_filter_trace_score.py # Tool 8
├── resampling_periodicity_score.py # Tool 9
├── dct_highpass_filter.py       # Tool 10
├── sobel_filter.py              # Tool 11
├── highpass_filter.py           # Tool 12
├── temp_ela.jpg                 # Demo image
README.md                    # This file
```


## 🔧 Technical Details

- **Input**: PIL Image, NumPy array, or file path (string)
- **Output**: NumPy array (float32)
- **Grayscale conversion**: 0.299×R + 0.587×G + 0.114×B
- **Boundary handling**: Symmetric padding (reflection)
- **All tools are self-contained** with no cross-dependencies

**Note**: All tools include comprehensive docstrings following NumPy documentation standards. Use `help(function_name)` for detailed information on any tool.