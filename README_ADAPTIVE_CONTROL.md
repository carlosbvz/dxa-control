# Adaptive Control System for DXA Image Processing Pipeline

## Overview

This module implements an **adaptive control system** using iterative optimization (grid search) to automatically adjust Kalman filter and contrast enhancement parameters for optimal image quality (PSNR, SSIM). This represents a modern control approach that adapts to each image's characteristics, eliminating manual parameter tuning and improving reproducibility.

## Features

### 🎯 **Adaptive Parameter Optimization**
- **Grid Search**: Systematically explores parameter combinations
- **Quality Metrics**: Optimizes PSNR and SSIM for each image
- **Automatic Selection**: Chooses best parameters automatically
- **Comprehensive Reporting**: Detailed analysis of all iterations

### 🔧 **Parameter Optimization**
- **Kalman Filter Parameters**:
  - Process Noise (Q): [5, 10, 20, 30, 50]
  - Measurement Noise (R): [0.001, 0.01, 0.1, 1.0]
  - Initial Uncertainty (P0): [1, 5, 10, 20]

- **Contrast Enhancement Parameters**:
  - CLAHE Clip Limit: [1.0, 2.0, 3.0, 4.0, 5.0]
  - Tile Grid Size: [(4,4), (8,8), (16,16)]

### 📊 **Comprehensive Reporting**
- **Iteration Results**: All parameter combinations tested
- **Quality Metrics**: PSNR, SSIM for each iteration
- **Visual Comparison**: Images from each iteration
- **Statistical Analysis**: Min, max, average metrics
- **Top Results**: Best 5 parameter combinations

## Usage

### Command Line Interface

```bash
# Classic pipeline (fixed parameters)
python src/main.py input_image.png output_dir

# Adaptive control pipeline (optimized parameters)
python src/main.py input_image.png output_dir --adaptive

# Batch processing with adaptive control
python src/main.py input_dir output_dir --batch --adaptive

# With custom configuration
python src/main.py input_image.png output_dir --adaptive --config config.json
```

### Python API

```python
from src.main import DXAPipeline

# Classic pipeline
pipeline_classic = DXAPipeline(use_adaptive_control=False)
results_classic = pipeline_classic.process_single_image("input.png", "output")

# Adaptive control pipeline
pipeline_adaptive = DXAPipeline(use_adaptive_control=True)
results_adaptive = pipeline_adaptive.process_single_image("input.png", "output")
```

### Demonstration Script

```bash
# Run the demonstration comparing classic vs adaptive pipelines
python demo_adaptive_control.py
```

## Configuration

### Default Configuration

```python
{
    'adaptive_control': {
        'max_iterations': 50,
        'kalman_ranges': {
            'process_noise': [5, 10, 20, 30, 50],
            'measurement_noise': [0.001, 0.01, 0.1, 1.0],
            'initial_uncertainty': [1, 5, 10, 20]
        },
        'contrast_ranges': {
            'clip_limit': [1.0, 2.0, 3.0, 4.0, 5.0],
            'tile_grid_size': [(4, 4), (8, 8), (16, 16)]
        }
    }
}
```

### Custom Configuration

Create a JSON file with your custom parameter ranges:

```json
{
    "adaptive_control": {
        "max_iterations": 30,
        "kalman_ranges": {
            "process_noise": [10, 20, 30],
            "measurement_noise": [0.01, 0.1],
            "initial_uncertainty": [5, 10]
        },
        "contrast_ranges": {
            "clip_limit": [2.0, 3.0, 4.0],
            "tile_grid_size": [(8, 8), (16, 16)]
        }
    }
}
```

## Output Structure

When using adaptive control, the output directory contains:

```
output_dir/
├── optimization_results/
│   ├── iteration_001/
│   │   ├── kalman_filtered.png
│   │   ├── contrast_enhanced.png
│   │   ├── roi_mask.png
│   │   └── parameters.json
│   ├── iteration_002/
│   │   └── ...
│   ├── optimization_report.md
│   └── detailed_results.json
├── kalman_filtered.png      # Best result
├── contrast_enhanced.png    # Best result
├── roi_mask.png            # Best result
└── summary.txt             # Processing summary
```

## Key Components

### 1. AdaptiveControlSystem Class

**Location**: `src/adaptive_control.py`

**Purpose**: Implements the grid search optimization algorithm

**Key Methods**:
- `optimize_parameters()`: Main optimization function
- `_generate_parameter_combinations()`: Creates parameter combinations
- `_save_iteration_results()`: Saves intermediate results
- `_generate_optimization_report()`: Creates comprehensive reports

### 2. OptimizationResult Dataclass

**Purpose**: Stores results for each parameter combination

**Fields**:
- `kalman_params`: Kalman filter parameters used
- `contrast_params`: Contrast enhancement parameters used
- `psnr`, `ssim`: Quality metrics
- `processing_time`: Time for this iteration
- `kalman_image`, `enhanced_image`, `roi_mask`: Processed images

### 3. Modified DXAPipeline

**Location**: `src/main.py`

**Changes**:
- Added `use_adaptive_control` parameter
- Integrated adaptive control system
- Enhanced reporting with optimization results
- Added `--adaptive` command line argument

## Quality Metrics

### PSNR (Peak Signal-to-Noise Ratio)
- **Range**: 0 to ∞ dB (higher is better)
- **Target**: ≥ 30 dB for good quality
- **Formula**: 20 * log10(MAX / sqrt(MSE))

### SSIM (Structural Similarity Index)
- **Range**: 0 to 1 (higher is better)
- **Target**: ≥ 0.85 for good quality
- **Measures**: Structural similarity between images

## Modern Control Principles

This implementation demonstrates several key principles of modern control:

### 1. **Adaptive Control**
- Parameters automatically adjust to each image's characteristics
- No manual tuning required
- System learns optimal parameters for each input

### 2. **Optimization-Based Control**
- Uses systematic search (grid search) to find optimal parameters
- Objective function: Maximize PSNR + 0.1*SSIM
- Multi-variable optimization (5 Kalman + 3 CLAHE parameters)

### 3. **Experimental Validation**
- Comprehensive testing of parameter combinations
- Statistical analysis of results
- Detailed documentation of optimization process

### 4. **Automated Decision Making**
- Automatic selection of best parameters
- No human intervention required
- Reproducible results

## Performance Considerations

### Computation Time
- **Default**: Up to 50 iterations
- **Typical Time**: 2-5 minutes per image
- **Configurable**: Adjust `max_iterations` parameter

### Memory Usage
- **Intermediate Storage**: All iteration results saved
- **Disk Space**: ~10-50 MB per image (depending on iterations)
- **RAM**: Minimal impact (processes one iteration at a time)

## Troubleshooting

### Common Issues

1. **Long Processing Time**
   - Reduce `max_iterations` in configuration
   - Use fewer parameter values in ranges

2. **Memory Issues**
   - Process images one at a time
   - Clean up intermediate results if needed

3. **Poor Quality Results**
   - Check parameter ranges are appropriate
   - Verify input image quality
   - Review optimization report for insights

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Course Alignment (SP-2159)

This implementation directly addresses SP-2159 course objectives:

### ✅ **Modern Control Techniques**
- Adaptive control systems
- Parameter optimization
- Multi-variable control

### ✅ **Experimental Validation**
- Systematic parameter testing
- Statistical analysis
- Performance comparison

### ✅ **Automation and Efficiency**
- Eliminates manual parameter tuning
- Improves reproducibility
- Reduces processing time per image

### ✅ **Real-World Application**
- Medical image processing
- Quality optimization
- Clinical relevance

## Future Enhancements

### Potential Improvements
1. **Machine Learning Integration**: Use ML to predict optimal parameters
2. **Parallel Processing**: Test multiple parameter combinations simultaneously
3. **Advanced Optimization**: Implement genetic algorithms or Bayesian optimization
4. **Real-Time Adaptation**: Continuous parameter adjustment during processing

### Research Applications
1. **Clinical Validation**: Test on real DXA images
2. **Performance Analysis**: Compare with other optimization methods
3. **Parameter Sensitivity**: Analyze parameter impact on results

## Citation

If using this code in research:

```bibtex
@software{adaptive_control_dxa,
  title={Adaptive Control System for DXA Image Processing},
  author={Carlos Benavides},
  year={2024},
  course={SP-2159 Técnicas Modernas de Control},
  institution={Universidad de Costa Rica}
}
```

---

**Author**: Carlos Benavides  
**Course**: SP-2159 Técnicas Modernas de Control  
**Institution**: Universidad de Costa Rica  
**Date**: 2024 