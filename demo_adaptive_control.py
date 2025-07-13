#!/usr/bin/env python3
"""
Demonstration Script for Adaptive Control System
===============================================

This script demonstrates the adaptive control system for DXA image processing,
comparing the classic pipeline with fixed parameters against the adaptive
pipeline that optimizes parameters for each image.

Author: Carlos Benavides
Course: SP-2159 Técnicas Modernas de Control
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
import argparse

# Add src to path
sys.path.append('src')

from main import DXAPipeline
from utils import ImageUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_image(size=(256, 256), noise_level=0.1):
    """
    Create a test DXA-like image with bone structure and noise.
    
    Args:
        size: Image size (height, width)
        noise_level: Noise level to add
        
    Returns:
        Test image as numpy array
    """
    # Create base image with bone-like structure
    y, x = np.ogrid[:size[0], :size[1]]
    
    # Create bone regions (simulating femur and spine)
    bone_mask = np.zeros(size)
    
    # Femur-like structure (vertical)
    femur_center = size[1] // 2
    femur_width = size[1] // 8
    bone_mask[:, femur_center-femur_width:femur_center+femur_width] = 1
    
    # Spine-like structure (horizontal)
    spine_center = size[0] // 2
    spine_height = size[0] // 6
    bone_mask[spine_center-spine_height:spine_center+spine_height, :] = 1
    
    # Create image with bone structure
    image = bone_mask * 0.8 + 0.2  # Bone regions are brighter
    
    # Add noise
    noise = np.random.normal(0, noise_level, size)
    image = image + noise
    
    # Normalize to [0, 1]
    image = np.clip(image, 0, 1)
    
    return image


def compare_pipelines(test_image, output_dir):
    """
    Compare classic vs adaptive control pipelines.
    
    Args:
        test_image: Input test image
        output_dir: Output directory for results
    """
    logger.info("Starting pipeline comparison")
    
    # Create output directories
    classic_dir = os.path.join(output_dir, 'classic_pipeline')
    adaptive_dir = os.path.join(output_dir, 'adaptive_pipeline')
    os.makedirs(classic_dir, exist_ok=True)
    os.makedirs(adaptive_dir, exist_ok=True)
    
    # Save test image
    test_image_path = os.path.join(output_dir, 'test_image.png')
    ImageUtils.save_image(test_image, test_image_path)
    
    # Test 1: Classic Pipeline
    logger.info("Testing Classic Pipeline")
    classic_start = time.time()
    
    classic_pipeline = DXAPipeline(use_adaptive_control=False)
    classic_results = classic_pipeline.process_single_image(
        test_image_path, classic_dir, save_intermediate=True
    )
    
    classic_time = time.time() - classic_start
    
    # Test 2: Adaptive Control Pipeline
    logger.info("Testing Adaptive Control Pipeline")
    adaptive_start = time.time()
    
    adaptive_pipeline = DXAPipeline(use_adaptive_control=True)
    adaptive_results = adaptive_pipeline.process_single_image(
        test_image_path, adaptive_dir, save_intermediate=True
    )
    
    adaptive_time = time.time() - adaptive_start
    
    # Generate comparison report
    generate_comparison_report(classic_results, adaptive_results, 
                             classic_time, adaptive_time, output_dir)
    
    return classic_results, adaptive_results


def generate_comparison_report(classic_results, adaptive_results, 
                             classic_time, adaptive_time, output_dir):
    """
    Generate a comprehensive comparison report.
    
    Args:
        classic_results: Results from classic pipeline
        adaptive_results: Results from adaptive pipeline
        classic_time: Processing time for classic pipeline
        adaptive_time: Processing time for adaptive pipeline
        output_dir: Output directory
    """
    report = f"""
# Pipeline Comparison Report
===========================

## Processing Times
- Classic Pipeline: {classic_time:.2f} seconds
- Adaptive Pipeline: {adaptive_time:.2f} seconds
- Time Difference: {adaptive_time - classic_time:.2f} seconds

## Quality Metrics Comparison

### Classic Pipeline:
"""
    
    if 'evaluation' in classic_results:
        classic_metrics = classic_results['evaluation']['metrics']
        report += f"""
- PSNR: {classic_metrics['psnr']:.2f} dB
- SSIM: {classic_metrics['ssim']:.4f}
- MSE: {classic_metrics['mse']:.6f}
"""
    
    report += """
### Adaptive Pipeline:
"""
    
    if 'evaluation' in adaptive_results:
        adaptive_metrics = adaptive_results['evaluation']['metrics']
        report += f"""
- PSNR: {adaptive_metrics['psnr']:.2f} dB
- SSIM: {adaptive_metrics['ssim']:.4f}
- MSE: {adaptive_metrics['mse']:.6f}
"""
    
    if 'optimization' in adaptive_results:
        opt = adaptive_results['optimization']
        report += f"""
## Optimization Results:
- Best PSNR Found: {opt['best_psnr']:.2f} dB
- Best SSIM Found: {opt['best_ssim']:.4f}
- Optimization Time: {opt['processing_time']:.2f} seconds
- Best Kalman Parameters: {opt['best_kalman_params']}
- Best Contrast Parameters: {opt['best_contrast_params']}
"""
    
    # Calculate improvements
    if 'evaluation' in classic_results and 'evaluation' in adaptive_results:
        classic_psnr = classic_results['evaluation']['metrics']['psnr']
        adaptive_psnr = adaptive_results['evaluation']['metrics']['psnr']
        classic_ssim = classic_results['evaluation']['metrics']['ssim']
        adaptive_ssim = adaptive_results['evaluation']['metrics']['ssim']
        
        psnr_improvement = adaptive_psnr - classic_psnr
        ssim_improvement = adaptive_ssim - classic_ssim
        
        report += f"""
## Improvements:
- PSNR Improvement: {psnr_improvement:+.2f} dB
- SSIM Improvement: {ssim_improvement:+.4f}
- Relative PSNR Improvement: {(psnr_improvement/classic_psnr)*100:+.1f}%
- Relative SSIM Improvement: {(ssim_improvement/classic_ssim)*100:+.1f}%
"""
    
    report += f"""
## File Structure:
```
{output_dir}/
├── test_image.png
├── classic_pipeline/
│   ├── kalman_filtered.png
│   ├── contrast_enhanced.png
│   ├── roi_mask.png
│   └── summary.txt
└── adaptive_pipeline/
    ├── optimization_results/
    │   ├── iteration_001/
    │   ├── iteration_002/
    │   ├── ...
    │   ├── optimization_report.md
    │   └── detailed_results.json
    ├── kalman_filtered.png
    ├── contrast_enhanced.png
    ├── roi_mask.png
    └── summary.txt
```

## Conclusion:
The adaptive control system demonstrates modern control principles by:
1. Automatically adapting parameters to each image's characteristics
2. Optimizing quality metrics (PSNR, SSIM) through systematic search
3. Providing comprehensive documentation of the optimization process
4. Eliminating manual parameter tuning while improving results

This approach aligns with SP-2159 course objectives by implementing:
- Adaptive control systems
- Parameter optimization
- Experimental validation
- Automated decision making
"""
    
    # Save report
    report_path = os.path.join(output_dir, 'comparison_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Comparison report saved to {report_path}")


def main():
    """Main demonstration function."""
    logger.info("Starting Adaptive Control Demonstration")

    parser = argparse.ArgumentParser(description="Demo Adaptive Control: Classic vs Adaptive pipeline")
    parser.add_argument('--input', type=str, help="Ruta a la imagen real (ej: Paciente01_HE.png)")
    parser.add_argument('--output', type=str, default="demo_results", help="Directorio de salida")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    if args.input:
        # Usar imagen real proporcionada
        logger.info(f"Usando imagen real: {args.input}")
        test_image_path = args.input
        # Cargar imagen como array numpy normalizado
        test_image = ImageUtils.load_image(test_image_path, normalize=True)
        # Guardar copia en el output_dir para consistencia
        ImageUtils.save_image(test_image, os.path.join(output_dir, 'test_image.png'))
    else:
        # Generar imagen sintética
        logger.info("Creating test DXA image")
        test_image = create_test_image(size=(256, 256), noise_level=0.15)
        ImageUtils.save_image(test_image, os.path.join(output_dir, 'test_image.png'))

    # Ejecutar comparación
    logger.info("Running pipeline comparison")
    compare_pipelines(test_image, output_dir)

    logger.info("Demonstration completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("Check the comparison_report.md file for detailed analysis")


if __name__ == "__main__":
    main() 