"""
Adaptive Control Module for DXA Image Processing Pipeline
=======================================================

This module implements an adaptive control system using iterative optimization
(grid search) to automatically adjust Kalman filter and contrast enhancement
parameters for optimal image quality (PSNR, SSIM).

This represents a modern control approach that adapts to each image's
characteristics, eliminating manual parameter tuning and improving
reproducibility and efficiency.

Author: Carlos Benavides
Course: SP-2159 Técnicas Modernas de Control
"""

import numpy as np
import cv2
import os
import time
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import itertools
import json
from dataclasses import dataclass

# Import project modules
from kalman_filter import KalmanFilterImage, apply_kalman_filter
from contrast_enhancement import ContrastEnhancer, enhance_contrast
from roi_tracking import detect_roi
from evaluation import ImageEvaluator
from utils import ImageUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Data class to store optimization results for each parameter combination."""
    kalman_params: Dict
    contrast_params: Dict
    psnr: float
    ssim: float
    processing_time: float
    iteration: int
    kalman_image: np.ndarray
    enhanced_image: np.ndarray
    roi_mask: np.ndarray


class AdaptiveControlSystem:
    """
    Adaptive control system using grid search optimization.
    
    This class implements a modern control approach that automatically
    optimizes pipeline parameters for each image to maximize quality metrics.
    """
    
    def __init__(self, 
                 kalman_param_ranges: Optional[Dict] = None,
                 contrast_param_ranges: Optional[Dict] = None,
                 max_iterations: int = 50):
        """
        Initialize the adaptive control system.
        
        Args:
            kalman_param_ranges: Ranges for Kalman filter parameters
            contrast_param_ranges: Ranges for contrast enhancement parameters
            max_iterations: Maximum number of iterations to limit computation time
        """
        self.max_iterations = max_iterations
        self.evaluator = ImageEvaluator()
        
        # Define parameter ranges for grid search
        self.kalman_ranges = kalman_param_ranges or {
            'process_noise': [5, 10, 20, 30, 50],
            'measurement_noise': [0.001, 0.01, 0.1, 1.0],
            'initial_uncertainty': [1, 5, 10, 20]
        }
        
        self.contrast_ranges = contrast_param_ranges or {
            'clip_limit': [1.0, 2.0, 3.0, 4.0, 5.0],
            'tile_grid_size': [(4, 4), (8, 8), (16, 16)]
        }
        
        logger.info("Adaptive control system initialized")
        logger.info(f"Kalman parameter combinations: {self._count_kalman_combinations()}")
        logger.info(f"Contrast parameter combinations: {self._count_contrast_combinations()}")
    
    def _count_kalman_combinations(self) -> int:
        """Count total Kalman parameter combinations."""
        return (len(self.kalman_ranges['process_noise']) *
                len(self.kalman_ranges['measurement_noise']) *
                len(self.kalman_ranges['initial_uncertainty']))
    
    def _count_contrast_combinations(self) -> int:
        """Count total contrast parameter combinations."""
        return (len(self.contrast_ranges['clip_limit']) *
                len(self.contrast_ranges['tile_grid_size']))
    
    def _generate_parameter_combinations(self) -> List[Tuple[Dict, Dict]]:
        """
        Generate all parameter combinations for grid search.
        
        Returns:
            List of (kalman_params, contrast_params) tuples
        """
        # Generate Kalman parameter combinations
        kalman_combinations = list(itertools.product(
            self.kalman_ranges['process_noise'],
            self.kalman_ranges['measurement_noise'],
            self.kalman_ranges['initial_uncertainty']
        ))
        
        # Generate contrast parameter combinations
        contrast_combinations = list(itertools.product(
            self.contrast_ranges['clip_limit'],
            self.contrast_ranges['tile_grid_size']
        ))
        
        # Combine all combinations
        all_combinations = []
        for kalman_params in kalman_combinations:
            for contrast_params in contrast_combinations:
                kalman_dict = {
                    'process_noise': kalman_params[0],
                    'measurement_noise': kalman_params[1],
                    'initial_uncertainty': kalman_params[2]
                }
                contrast_dict = {
                    'clip_limit': contrast_params[0],
                    'tile_grid_size': contrast_params[1]
                }
                all_combinations.append((kalman_dict, contrast_dict))
        
        # Limit to max_iterations if needed
        if len(all_combinations) > self.max_iterations:
            logger.warning(f"Limiting combinations from {len(all_combinations)} to {self.max_iterations}")
            all_combinations = all_combinations[:self.max_iterations]
        
        return all_combinations
    
    def optimize_parameters(self, 
                          original_image: np.ndarray,
                          output_dir: str) -> Tuple[Dict, Dict, OptimizationResult]:
        """
        Optimize parameters using grid search.
        
        Args:
            original_image: Input image to optimize for
            output_dir: Directory to save intermediate results
            
        Returns:
            Tuple of (best_kalman_params, best_contrast_params, best_result)
        """
        logger.info("Starting parameter optimization using grid search")
        start_time = time.time()
        
        # Create output directory for optimization results
        optimization_dir = os.path.join(output_dir, 'optimization_results')
        os.makedirs(optimization_dir, exist_ok=True)
        
        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_combinations()
        logger.info(f"Testing {len(parameter_combinations)} parameter combinations")
        
        # Store all results
        all_results = []
        best_result = None
        best_score = -float('inf')
        
        # Test each parameter combination
        for i, (kalman_params, contrast_params) in enumerate(parameter_combinations):
            logger.info(f"Iteration {i+1}/{len(parameter_combinations)}")
            logger.info(f"Testing Kalman: {kalman_params}, Contrast: {contrast_params}")
            
            try:
                # Process image with current parameters
                iteration_start = time.time()
                
                # Step 1: Apply Kalman filter
                kalman_filter = KalmanFilterImage(**kalman_params)
                kalman_image = kalman_filter.filter_image(original_image)
                
                # Step 2: Apply contrast enhancement
                enhanced_image = enhance_contrast(
                    kalman_image,
                    method='clahe',
                    **contrast_params
                )
                
                # Step 3: Detect ROI (for completeness)
                roi_mask = detect_roi(enhanced_image, method='otsu')
                
                # Step 4: Calculate quality metrics
                psnr = self.evaluator.calculate_psnr(original_image, enhanced_image)
                ssim = self.evaluator.calculate_ssim(original_image, enhanced_image)
                
                processing_time = time.time() - iteration_start
                
                # Create result object
                result = OptimizationResult(
                    kalman_params=kalman_params,
                    contrast_params=contrast_params,
                    psnr=psnr,
                    ssim=ssim,
                    processing_time=processing_time,
                    iteration=i+1,
                    kalman_image=kalman_image,
                    enhanced_image=enhanced_image,
                    roi_mask=roi_mask
                )
                
                all_results.append(result)
                
                # Save intermediate results
                self._save_iteration_results(result, optimization_dir, i+1)
                
                # Update best result (using PSNR as primary metric, SSIM as secondary)
                current_score = psnr + 0.1 * ssim  # Weighted combination
                if current_score > best_score:
                    best_score = current_score
                    best_result = result
                    logger.info(f"New best result: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")
                
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {e}")
                continue
        
        # Generate optimization report
        self._generate_optimization_report(all_results, best_result, optimization_dir)
        
        total_time = time.time() - start_time
        logger.info(f"Optimization completed in {total_time:.2f} seconds")
        logger.info(f"Best parameters found: Kalman={best_result.kalman_params}, Contrast={best_result.contrast_params}")
        logger.info(f"Best metrics: PSNR={best_result.psnr:.2f}dB, SSIM={best_result.ssim:.4f}")
        
        return best_result.kalman_params, best_result.contrast_params, best_result
    
    def _save_iteration_results(self, 
                               result: OptimizationResult,
                               output_dir: str,
                               iteration: int) -> None:
        """
        Save results for a single iteration.
        
        Args:
            result: Optimization result
            output_dir: Output directory
            iteration: Iteration number
        """
        iteration_dir = os.path.join(output_dir, f'iteration_{iteration:03d}')
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Save images
        ImageUtils.save_image(result.kalman_image, 
                            os.path.join(iteration_dir, 'kalman_filtered.png'))
        ImageUtils.save_image(result.enhanced_image, 
                            os.path.join(iteration_dir, 'contrast_enhanced.png'))
        ImageUtils.save_image(result.roi_mask.astype(np.float32), 
                            os.path.join(iteration_dir, 'roi_mask.png'))
        
        # Save parameters and metrics
        iteration_data = {
            'iteration': iteration,
            'kalman_params': result.kalman_params,
            'contrast_params': result.contrast_params,
            'psnr': result.psnr,
            'ssim': result.ssim,
            'processing_time': result.processing_time
        }
        
        with open(os.path.join(iteration_dir, 'parameters.json'), 'w') as f:
            json.dump(iteration_data, f, indent=2)
    
    def _generate_optimization_report(self, 
                                    all_results: List[OptimizationResult],
                                    best_result: OptimizationResult,
                                    output_dir: str) -> None:
        """
        Generate comprehensive optimization report.
        
        Args:
            all_results: All optimization results
            best_result: Best result found
            output_dir: Output directory
        """
        # Create summary data
        psnr_values = [r.psnr for r in all_results]
        ssim_values = [r.ssim for r in all_results]
        processing_times = [r.processing_time for r in all_results]
        
        # Generate report
        report = f"""
# Adaptive Control Optimization Report
====================================

## Summary Statistics
- Total iterations: {len(all_results)}
- Best PSNR: {best_result.psnr:.2f} dB
- Best SSIM: {best_result.ssim:.4f}
- Average PSNR: {np.mean(psnr_values):.2f} dB
- Average SSIM: {np.mean(ssim_values):.4f}
- Total processing time: {np.sum(processing_times):.2f} seconds

## Best Parameters Found
### Kalman Filter Parameters:
- Process Noise (Q): {best_result.kalman_params['process_noise']}
- Measurement Noise (R): {best_result.kalman_params['measurement_noise']}
- Initial Uncertainty (P0): {best_result.kalman_params['initial_uncertainty']}

### Contrast Enhancement Parameters:
- CLAHE Clip Limit: {best_result.contrast_params['clip_limit']}
- Tile Grid Size: {best_result.contrast_params['tile_grid_size']}

## Parameter Analysis
### PSNR Statistics:
- Min: {np.min(psnr_values):.2f} dB
- Max: {np.max(psnr_values):.2f} dB
- Std: {np.std(psnr_values):.2f} dB

### SSIM Statistics:
- Min: {np.min(ssim_values):.4f}
- Max: {np.max(ssim_values):.4f}
- Std: {np.std(ssim_values):.4f}

## Top 5 Results
"""
        
        # Sort results by PSNR
        sorted_results = sorted(all_results, key=lambda x: x.psnr, reverse=True)
        
        for i, result in enumerate(sorted_results[:5]):
            report += f"""
### Rank {i+1}
- PSNR: {result.psnr:.2f} dB
- SSIM: {result.ssim:.4f}
- Kalman: {result.kalman_params}
- Contrast: {result.contrast_params}
"""
        
        # Save report
        report_path = os.path.join(output_dir, 'optimization_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save detailed results as JSON
        detailed_results = []
        for result in all_results:
            detailed_results.append({
                'iteration': result.iteration,
                'kalman_params': result.kalman_params,
                'contrast_params': result.contrast_params,
                'psnr': result.psnr,
                'ssim': result.ssim,
                'processing_time': result.processing_time
            })
        
        json_path = os.path.join(output_dir, 'detailed_results.json')
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info(f"Optimization report saved to {output_dir}")


def create_adaptive_control_system(config: Optional[Dict] = None) -> AdaptiveControlSystem:
    """
    Factory function to create an adaptive control system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured adaptive control system
    """
    if config is None:
        return AdaptiveControlSystem()
    
    kalman_ranges = config.get('kalman_ranges')
    contrast_ranges = config.get('contrast_ranges')
    max_iterations = config.get('max_iterations', 50)
    
    return AdaptiveControlSystem(
        kalman_param_ranges=kalman_ranges,
        contrast_param_ranges=contrast_ranges,
        max_iterations=max_iterations
    ) 