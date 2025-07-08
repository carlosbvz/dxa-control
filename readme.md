# DXA Image Processing Pipeline

## Filtrado de Kalman para Reducción de Ruido y Mejora de Contraste en Imágenes DXA: Un Enfoque de Estimación Bayesiana

**Author:** Carlos Benavides  
**Course:** SP-2141 Teoría de la detección y estimación  
**Institution:** Universidad de Costa Rica

## Project Overview

This project implements a comprehensive DXA (Dual-energy X-ray Absorptiometry) image processing pipeline that applies concepts from signal detection and estimation theory. The system uses Kalman filtering for noise reduction and contrast enhancement, followed by ROI detection and BMD (Bone Mineral Density) extraction.

### Key Features

- **Kalman Filtering**: Spatial Kalman filter for noise reduction and signal estimation
- **Contrast Enhancement**: Multiple enhancement techniques (CLAHE, histogram equalization, etc.)
- **ROI Detection**: Automatic detection of bone regions using various algorithms
- **BMD Extraction**: Statistical analysis of bone mineral density
- **Quality Evaluation**: Comprehensive metrics for processing quality assessment
- **Batch Processing**: Support for processing multiple images

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download the project files
2. Navigate to the project directory:
   ```bash
   cd "./code"
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
code/
├── data/
│   ├── raw/          # Original DXA images (HE/LE)
│   └── masks/        # Manual masks (if available)
├── results/          # Processing outputs
├── src/
│   ├── kalman_filter.py      # Kalman filtering implementation
│   ├── contrast_enhancement.py # Contrast enhancement methods
│   ├── roi_tracking.py       # ROI detection algorithms
│   ├── bmd_extraction.py     # BMD analysis
│   ├── evaluation.py         # Quality evaluation metrics
│   ├── utils.py             # Utility functions
│   └── main.py              # Main pipeline orchestration
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks for examples
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Usage

### Command Line Interface

#### Single Image Processing

```bash
python src/main.py data/raw/Paciente01_HE.png results/   
```

#### Batch Processing

```bash
python src/main.py input_directory output_directory --batch --pattern "*.png"
```

#### With Custom Configuration

```bash
python src/main.py input_image.png output_directory --config config.json
```

### Python API

#### Basic Usage

```python
from src.main import DXAPipeline

# Initialize pipeline
pipeline = DXAPipeline()

# Process single image
results = pipeline.process_single_image('input.png', 'output_dir')

# Process batch
results = pipeline.process_batch('input_dir', 'output_dir')
```

#### Individual Components

```python
from src.kalman_filter import KalmanFilterImage
from src.contrast_enhancement import ContrastEnhancer
from src.roi_tracking import ROITracker
from src.bmd_extraction import BMDExtractor

# Kalman filtering
kf = KalmanFilterImage()
filtered_image = kf.filter_image(image)

# Contrast enhancement
enhancer = ContrastEnhancer()
enhanced_image = enhancer.adaptive_histogram_equalization(filtered_image)

# ROI detection
tracker = ROITracker()
roi_mask = tracker.detect_bone_regions(enhanced_image, method='otsu')

# BMD extraction
extractor = BMDExtractor()
bmd_stats = extractor.extract_bmd_values(enhanced_image, roi_mask)
```

## Configuration

The pipeline can be configured using a JSON configuration file:

```json
{
  "kalman": {
    "process_noise": 0.01,
    "measurement_noise": 0.1,
    "initial_uncertainty": 0.1
  },
  "contrast": {
    "method": "clahe",
    "clip_limit": 2.0,
    "tile_grid_size": [8, 8]
  },
  "roi": {
    "method": "otsu",
    "post_process": true
  },
  "bmd": {
    "method": "direct"
  },
  "evaluation": {
    "calculate_metrics": true,
    "save_results": true
  }
}
```

## Processing Pipeline

The complete processing pipeline consists of five main steps:

1. **Image Loading & Preprocessing**
   - Load DXA image
   - Normalize pixel values
   - Validate image properties

2. **Kalman Filtering**
   - Apply spatial Kalman filter
   - Reduce noise while preserving structure
   - Estimate true pixel values

3. **Contrast Enhancement**
   - Apply CLAHE or other enhancement methods
   - Improve image visibility
   - Enhance bone-tissue contrast

4. **ROI Detection**
   - Detect bone regions using thresholding or clustering
   - Apply morphological post-processing
   - Extract ROI properties

5. **BMD Analysis**
   - Extract BMD values from detected regions
   - Calculate statistical measures
   - Perform distribution analysis

6. **Quality Evaluation**
   - Calculate PSNR, SSIM, MSE metrics
   - Compare with reference image
   - Generate evaluation report

## Output Files

The pipeline generates several output files:

- `kalman_filtered.png`: Image after Kalman filtering
- `contrast_enhanced.png`: Image after contrast enhancement
- `roi_mask.png`: Binary mask of detected bone regions
- `processing_results.json`: Complete processing results
- `summary.txt`: Human-readable summary
- `batch_summary.txt`: Summary for batch processing

## Testing

Run the unit tests:

```bash
python -m pytest tests/
```

Or run individual test files:

```bash
python tests/test_kalman_filter.py
```

## Examples

### Jupyter Notebook

See `notebooks/example_usage.ipynb` for a comprehensive example with visualizations.

### Basic Example

```python
import numpy as np
from src.main import DXAPipeline
from src.utils import ImageUtils

# Create test image
test_image = ImageUtils.create_test_image((200, 200), noise_level=0.2)
ImageUtils.save_image(test_image, 'test_input.png')

# Process image
pipeline = DXAPipeline()
results = pipeline.process_single_image('test_input.png', 'test_output')

print(f"Processing time: {results['processing_time']:.2f} seconds")
print(f"Mean BMD: {results['bmd']['statistics']['mean_bmd']:.4f}")
```

## Theoretical Background

This project applies concepts from SP-2141 (Teoría de la detección y estimación):

### Estimation Theory
- **Bayesian Estimation**: Kalman filter uses Bayesian principles for state estimation
- **Maximum Likelihood**: Parameter estimation for noise characteristics
- **Recursive Estimation**: Sequential processing of image pixels

### Detection Theory
- **Binary Classification**: Bone vs. soft tissue detection
- **Threshold Detection**: Otsu and adaptive thresholding methods
- **Signal Detection**: ROI detection in noisy images

### Signal Processing
- **Noise Reduction**: Kalman filtering for spatial noise reduction
- **Contrast Enhancement**: Various techniques for signal enhancement
- **Quality Metrics**: PSNR, SSIM, and other evaluation measures

## Performance Metrics

The pipeline aims to achieve:

- **PSNR ≥ 30 dB**: Peak Signal-to-Noise Ratio
- **SSIM ≥ 0.85**: Structural Similarity Index
- **Processing Time ≤ 5 seconds**: Per image
- **ROI Detection Accuracy ≥ 85%**: For bone region detection

## Limitations

- Currently supports only 2D images
- Optimized for DXA images specifically
- Requires sufficient image quality for reliable ROI detection
- Processing time scales with image size

## Future Work

- Support for 3D image processing
- Integration with deep learning methods
- Real-time processing capabilities
- Additional BMD analysis techniques
- Web-based interface

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is developed for academic purposes as part of the SP-2141 course at Universidad de Costa Rica.

## Contact

**Author:** Carlos Benavides  
**Email:** carlos.benavidesviquez@ucr.ac.cr
**Course:** SP-2141 Teoría de la detección y estimación  
**Institution:** Universidad de Costa Rica

## Acknowledgments

- Professor and teaching staff of SP-2141
- Open source libraries (OpenCV, NumPy, SciPy, scikit-image)
- Research community in medical image processing

# Navigate to the project directory
cd "Maestria/cursos/Sem2/SP-2141 Teoría de la detección y estimación/proyecto/code"

# Install dependencies
pip install -r requirements.txt

# Test the pipeline
python test_pipeline.py

# Process a single image
python src/main.py data/raw/Paciente01_HE.png results/

# Process all images in batch
python src/main.py data/raw/ results/ --batch --pattern "*.png"