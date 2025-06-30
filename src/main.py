"""
Main Module for DXA Image Processing Pipeline
============================================

This module orchestrates the complete DXA image processing pipeline,
combining Kalman filtering, contrast enhancement, ROI detection, and BMD extraction.

Author: Carlos Benavides
Course: SP-2141 Teoría de la detección y estimación
"""

import numpy as np
import cv2
import os
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

# Import project modules
from kalman_filter import KalmanFilterImage, apply_kalman_filter
from contrast_enhancement import ContrastEnhancer, enhance_contrast
from roi_tracking import ROITracker, detect_roi
from bmd_extraction import BMDExtractor, extract_bmd
from evaluation import ImageEvaluator, evaluate_image_quality
from utils import ImageUtils, DataUtils, ValidationUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DXAPipeline:
    """
    Complete DXA image processing pipeline.
    
    This class orchestrates the entire processing workflow from image loading
    to final BMD analysis and evaluation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DXA processing pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.kalman_filter = KalmanFilterImage(**self.config.get('kalman', {}))
        self.contrast_enhancer = ContrastEnhancer()
        self.roi_tracker = ROITracker()
        self.bmd_extractor = BMDExtractor()
        self.evaluator = ImageEvaluator()
        
        logger.info("DXA processing pipeline initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'kalman': {
                'process_noise': 0.01,
                'measurement_noise': 0.1,
                'initial_uncertainty': 0.1
            },
            'contrast': {
                'method': 'clahe',
                'clip_limit': 2.0,
                'tile_grid_size': (8, 8)
            },
            'roi': {
                'method': 'otsu',
                'post_process': True
            },
            'bmd': {
                'method': 'direct'
            },
            'evaluation': {
                'calculate_metrics': True,
                'save_results': True
            }
        }
    
    def process_single_image(self, 
                           input_path: str,
                           output_dir: str,
                           save_intermediate: bool = True) -> Dict:
        """
        Process a single DXA image through the complete pipeline.
        
        Args:
            input_path: Path to input image
            output_dir: Output directory for results
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary containing all processing results
        """
        logger.info(f"Processing image: {input_path}")
        start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and validate image
        try:
            original_image = ImageUtils.load_image(input_path, normalize=True)
            ValidationUtils.validate_image(original_image)
        except Exception as e:
            logger.error(f"Failed to load image {input_path}: {e}")
            raise
        
        results = {
            'input_path': input_path,
            'output_dir': output_dir,
            'processing_time': 0,
            'image_info': ImageUtils.calculate_image_statistics(original_image)
        }
        
        # Step 1: Kalman Filtering
        logger.info("Step 1: Applying Kalman filter")
        kalman_start = time.time()
        
        filtered_image = self.kalman_filter.filter_image(original_image)
        
        if save_intermediate:
            kalman_path = os.path.join(output_dir, 'kalman_filtered.png')
            ImageUtils.save_image(filtered_image, kalman_path)
        
        results['kalman'] = {
            'filtered_image': filtered_image,
            'processing_time': time.time() - kalman_start
        }
        
        # Step 2: Contrast Enhancement
        logger.info("Step 2: Enhancing contrast")
        contrast_start = time.time()
        
        enhanced_image = enhance_contrast(
            filtered_image, 
            method=self.config['contrast']['method'],
            **self.config['contrast']
        )
        
        if save_intermediate:
            enhanced_path = os.path.join(output_dir, 'contrast_enhanced.png')
            ImageUtils.save_image(enhanced_image, enhanced_path)
        
        results['contrast'] = {
            'enhanced_image': enhanced_image,
            'processing_time': time.time() - contrast_start
        }
        
        # Step 3: ROI Detection
        logger.info("Step 3: Detecting regions of interest")
        roi_start = time.time()
        
        roi_mask = detect_roi(
            enhanced_image,
            method=self.config['roi']['method']
        )
        
        if save_intermediate:
            roi_path = os.path.join(output_dir, 'roi_mask.png')
            ImageUtils.save_image(roi_mask.astype(np.float32), roi_path)
        
        results['roi'] = {
            'mask': roi_mask,
            'properties': self.roi_tracker.extract_roi_properties(enhanced_image, roi_mask),
            'processing_time': time.time() - roi_start
        }
        
        # Step 4: BMD Extraction
        logger.info("Step 4: Extracting BMD values")
        bmd_start = time.time()
        
        bmd_stats = extract_bmd(
            enhanced_image,
            roi_mask,
            method=self.config['bmd']['method']
        )
        
        results['bmd'] = {
            'statistics': bmd_stats,
            'processing_time': time.time() - bmd_start
        }
        
        # Step 5: Evaluation (if reference image available)
        if self.config['evaluation']['calculate_metrics']:
            logger.info("Step 5: Evaluating processing quality")
            eval_start = time.time()
            
            # Compare with original image
            eval_metrics = self.evaluator.evaluate_all_metrics(original_image, enhanced_image)
            
            results['evaluation'] = {
                'metrics': eval_metrics,
                'processing_time': time.time() - eval_start
            }
        
        # Calculate total processing time
        results['processing_time'] = time.time() - start_time
        
        # Save results
        if self.config['evaluation']['save_results']:
            results_path = os.path.join(output_dir, 'processing_results.json')
            DataUtils.save_results(results, results_path, format='json')
        
        # Generate summary
        self._generate_summary(results, output_dir)
        
        logger.info(f"Processing completed in {results['processing_time']:.2f} seconds")
        return results
    
    def process_batch(self, 
                     input_dir: str,
                     output_dir: str,
                     file_pattern: str = "*.png") -> List[Dict]:
        """
        Process multiple images in batch.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for results
            file_pattern: Pattern to match image files
            
        Returns:
            List of processing results
        """
        logger.info(f"Processing batch from {input_dir}")
        
        # Get list of input files
        input_files = DataUtils.get_file_list(input_dir, file_pattern)
        
        if not input_files:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        
        # Process each image
        all_results = []
        for i, input_file in enumerate(input_files):
            logger.info(f"Processing file {i+1}/{len(input_files)}: {input_file}")
            
            # Create output subdirectory
            file_name = Path(input_file).stem
            file_output_dir = os.path.join(output_dir, file_name)
            
            try:
                results = self.process_single_image(input_file, file_output_dir)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")
                continue
        
        # Generate batch summary
        self._generate_batch_summary(all_results, output_dir)
        
        logger.info(f"Batch processing completed: {len(all_results)}/{len(input_files)} successful")
        return all_results
    
    def _generate_summary(self, results: Dict, output_dir: str) -> None:
        """
        Generate processing summary.
        
        Args:
            results: Processing results
            output_dir: Output directory
        """
        summary = f"""
DXA Image Processing Summary
===========================

Input File: {results['input_path']}
Processing Time: {results['processing_time']:.2f} seconds

Image Information:
- Shape: {results['image_info']['shape']}
- Mean: {results['image_info']['mean']:.4f}
- Std: {results['image_info']['std']:.4f}

ROI Detection:
- Bone Area: {results['roi']['properties']['area']} pixels
- Centroid: {results['roi']['properties']['centroid']}

BMD Analysis:
- Mean BMD: {results['bmd']['statistics']['mean_bmd']:.4f}
- Std BMD: {results['bmd']['statistics']['std_bmd']:.4f}
- Total Pixels: {results['bmd']['statistics']['total_pixels']}
"""
        
        if 'evaluation' in results:
            eval_metrics = results['evaluation']['metrics']
            summary += f"""
Quality Metrics:
- PSNR: {eval_metrics['psnr']:.2f} dB
- SSIM: {eval_metrics['ssim']:.4f}
- MSE: {eval_metrics['mse']:.6f}
- SNR: {eval_metrics['snr']:.2f} dB
"""
        
        # Save summary
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"Summary saved to {summary_path}")
    
    def _generate_batch_summary(self, results: List[Dict], output_dir: str) -> None:
        """
        Generate batch processing summary.
        
        Args:
            results: List of processing results
            output_dir: Output directory
        """
        if not results:
            return
        
        # Calculate batch statistics
        processing_times = [r['processing_time'] for r in results]
        bmd_values = [r['bmd']['statistics']['mean_bmd'] for r in results]
        
        summary = f"""
Batch Processing Summary
=======================

Total Images Processed: {len(results)}
Average Processing Time: {np.mean(processing_times):.2f} seconds
Total Processing Time: {np.sum(processing_times):.2f} seconds

BMD Statistics:
- Mean BMD: {np.mean(bmd_values):.4f}
- Std BMD: {np.std(bmd_values):.4f}
- Min BMD: {np.min(bmd_values):.4f}
- Max BMD: {np.max(bmd_values):.4f}
"""
        
        if 'evaluation' in results[0]:
            psnr_values = [r['evaluation']['metrics']['psnr'] for r in results]
            ssim_values = [r['evaluation']['metrics']['ssim'] for r in results]
            
            summary += f"""
Quality Metrics (Average):
- PSNR: {np.mean(psnr_values):.2f} dB
- SSIM: {np.mean(ssim_values):.4f}
"""
        
        # Save batch summary
        batch_summary_path = os.path.join(output_dir, 'batch_summary.txt')
        with open(batch_summary_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"Batch summary saved to {batch_summary_path}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='DXA Image Processing Pipeline')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('output', help='Output directory')
    parser.add_argument('--batch', action='store_true', help='Process directory in batch')
    parser.add_argument('--pattern', default='*.png', help='File pattern for batch processing')
    parser.add_argument('--config', help='Configuration file (JSON)')
    parser.add_argument('--no-intermediate', action='store_true', help='Skip saving intermediate results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        config = DataUtils.load_results(args.config, format='json')
    
    # Initialize pipeline
    pipeline = DXAPipeline(config)
    
    # Process images
    if args.batch:
        # Batch processing
        if not os.path.isdir(args.input):
            logger.error("Input must be a directory for batch processing")
            return
        
        results = pipeline.process_batch(args.input, args.output, args.pattern)
        print(f"Processed {len(results)} images successfully")
    
    else:
        # Single image processing
        if not os.path.isfile(args.input):
            logger.error("Input must be a file for single image processing")
            return
        
        results = pipeline.process_single_image(
            args.input, 
            args.output, 
            save_intermediate=not args.no_intermediate
        )
        print("Image processed successfully")
    
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    # Example usage without command line arguments
    if len(os.sys.argv) == 1:
        # Create test pipeline
        pipeline = DXAPipeline()
        
        # Create test image
        test_image = ImageUtils.create_test_image((200, 200), noise_level=0.2)
        
        # Save test image
        test_input = "test_input.png"
        ImageUtils.save_image(test_image, test_input)
        
        # Process test image
        results = pipeline.process_single_image(test_input, "test_output")
        
        print("Test processing completed!")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"Mean BMD: {results['bmd']['statistics']['mean_bmd']:.4f}")
        
        # Clean up
        os.remove(test_input)
    else:
        main() 