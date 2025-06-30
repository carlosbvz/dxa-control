"""
Evaluation Module for DXA Image Processing
=========================================

This module implements evaluation metrics for assessing the quality
of processed DXA images, including PSNR, SSIM, MSE, and other metrics.

Author: Carlos Benavides
Course: SP-2141 Teoría de la detección y estimación
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import stats
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageEvaluator:
    """
    Image quality evaluation for DXA image processing.
    
    This class provides various metrics to assess the quality of
    processed images compared to reference images.
    """
    
    def __init__(self):
        """Initialize the image evaluator."""
        logger.info("Image evaluator initialized")
    
    def calculate_psnr(self, 
                      original: np.ndarray,
                      processed: np.ndarray,
                      max_val: float = 1.0) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            original: Original/reference image
            processed: Processed image
            max_val: Maximum possible pixel value
            
        Returns:
            PSNR value in dB
        """
        # Ensure images are in the same format
        original_norm = self._normalize_image(original)
        processed_norm = self._normalize_image(processed)
        
        # Calculate PSNR
        psnr_value = psnr(original_norm, processed_norm, data_range=max_val)
        
        logger.info(f"PSNR calculated: {psnr_value:.2f} dB")
        return psnr_value
    
    def calculate_ssim(self, 
                      original: np.ndarray,
                      processed: np.ndarray,
                      win_size: int = 11) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            original: Original/reference image
            processed: Processed image
            win_size: Window size for SSIM calculation
            
        Returns:
            SSIM value (0-1, higher is better)
        """
        # Ensure images are in the same format
        original_norm = self._normalize_image(original)
        processed_norm = self._normalize_image(processed)
        
        # Calculate SSIM
        ssim_value = ssim(original_norm, processed_norm, 
                         win_size=win_size, data_range=1.0)
        
        logger.info(f"SSIM calculated: {ssim_value:.4f}")
        return ssim_value
    
    def calculate_mse(self, 
                     original: np.ndarray,
                     processed: np.ndarray) -> float:
        """
        Calculate Mean Squared Error (MSE).
        
        Args:
            original: Original/reference image
            processed: Processed image
            
        Returns:
            MSE value
        """
        # Ensure images are in the same format
        original_norm = self._normalize_image(original)
        processed_norm = self._normalize_image(processed)
        
        # Calculate MSE
        mse_value = np.mean((original_norm - processed_norm) ** 2)
        
        logger.info(f"MSE calculated: {mse_value:.6f}")
        return mse_value
    
    def calculate_mae(self, 
                     original: np.ndarray,
                     processed: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error (MAE).
        
        Args:
            original: Original/reference image
            processed: Processed image
            
        Returns:
            MAE value
        """
        # Ensure images are in the same format
        original_norm = self._normalize_image(original)
        processed_norm = self._normalize_image(processed)
        
        # Calculate MAE
        mae_value = np.mean(np.abs(original_norm - processed_norm))
        
        logger.info(f"MAE calculated: {mae_value:.6f}")
        return mae_value
    
    def calculate_snr(self, 
                     original: np.ndarray,
                     processed: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR).
        
        Args:
            original: Original/reference image
            processed: Processed image
            
        Returns:
            SNR value in dB
        """
        # Ensure images are in the same format
        original_norm = self._normalize_image(original)
        processed_norm = self._normalize_image(processed)
        
        # Calculate signal and noise power
        signal_power = np.mean(original_norm ** 2)
        noise_power = np.mean((original_norm - processed_norm) ** 2)
        
        # Avoid division by zero
        if noise_power == 0:
            snr_value = float('inf')
        else:
            snr_value = 10 * np.log10(signal_power / noise_power)
        
        logger.info(f"SNR calculated: {snr_value:.2f} dB")
        return snr_value
    
    def calculate_correlation(self, 
                            original: np.ndarray,
                            processed: np.ndarray) -> float:
        """
        Calculate correlation coefficient between images.
        
        Args:
            original: Original/reference image
            processed: Processed image
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Ensure images are in the same format
        original_norm = self._normalize_image(original)
        processed_norm = self._normalize_image(processed)
        
        # Flatten images for correlation calculation
        original_flat = original_norm.flatten()
        processed_flat = processed_norm.flatten()
        
        # Calculate correlation
        correlation = np.corrcoef(original_flat, processed_flat)[0, 1]
        
        logger.info(f"Correlation calculated: {correlation:.4f}")
        return correlation
    
    def calculate_entropy(self, image: np.ndarray) -> float:
        """
        Calculate image entropy (information content).
        
        Args:
            image: Input image
            
        Returns:
            Entropy value
        """
        # Normalize image
        image_norm = self._normalize_image(image)
        
        # Convert to 8-bit for histogram calculation
        image_8bit = (image_norm * 255).astype(np.uint8)
        
        # Calculate histogram
        hist = cv2.calcHist([image_8bit], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Remove zero bins
        hist = hist[hist > 0]
        
        # Normalize histogram
        hist = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        logger.info(f"Entropy calculated: {entropy:.4f}")
        return entropy
    
    def calculate_contrast_improvement(self, 
                                     original: np.ndarray,
                                     processed: np.ndarray) -> float:
        """
        Calculate contrast improvement ratio.
        
        Args:
            original: Original image
            processed: Processed image
            
        Returns:
            Contrast improvement ratio
        """
        # Calculate contrast for both images
        original_contrast = np.std(original)
        processed_contrast = np.std(processed)
        
        # Calculate improvement ratio
        if original_contrast == 0:
            improvement = 0.0
        else:
            improvement = (processed_contrast - original_contrast) / original_contrast
        
        logger.info(f"Contrast improvement: {improvement:.4f}")
        return improvement
    
    def evaluate_all_metrics(self, 
                           original: np.ndarray,
                           processed: np.ndarray) -> Dict:
        """
        Calculate all evaluation metrics.
        
        Args:
            original: Original/reference image
            processed: Processed image
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic quality metrics
        metrics['psnr'] = self.calculate_psnr(original, processed)
        metrics['ssim'] = self.calculate_ssim(original, processed)
        metrics['mse'] = self.calculate_mse(original, processed)
        metrics['mae'] = self.calculate_mae(original, processed)
        metrics['snr'] = self.calculate_snr(original, processed)
        metrics['correlation'] = self.calculate_correlation(original, processed)
        
        # Image-specific metrics
        metrics['original_entropy'] = self.calculate_entropy(original)
        metrics['processed_entropy'] = self.calculate_entropy(processed)
        metrics['contrast_improvement'] = self.calculate_contrast_improvement(original, processed)
        
        # Additional statistics
        metrics['original_mean'] = np.mean(original)
        metrics['processed_mean'] = np.mean(processed)
        metrics['original_std'] = np.std(original)
        metrics['processed_std'] = np.std(processed)
        
        logger.info("All evaluation metrics calculated")
        return metrics
    
    def compare_methods(self, 
                       original: np.ndarray,
                       processed_images: Dict[str, np.ndarray]) -> Dict:
        """
        Compare multiple processing methods.
        
        Args:
            original: Original/reference image
            processed_images: Dictionary of processed images
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        for method_name, processed_image in processed_images.items():
            metrics = self.evaluate_all_metrics(original, processed_image)
            comparison[method_name] = metrics
        
        logger.info(f"Comparison completed for {len(processed_images)} methods")
        return comparison
    
    def generate_evaluation_report(self, 
                                 comparison_results: Dict,
                                 output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            comparison_results: Results from compare_methods
            output_file: Optional file to save report
            
        Returns:
            Report string
        """
        report = "DXA Image Processing Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Summary table
        report += "Method Comparison Summary:\n"
        report += "-" * 30 + "\n"
        
        # Header
        methods = list(comparison_results.keys())
        metrics = ['psnr', 'ssim', 'mse', 'snr']
        
        report += f"{'Method':<15}"
        for metric in metrics:
            report += f"{metric.upper():<10}"
        report += "\n"
        
        # Data rows
        for method in methods:
            report += f"{method:<15}"
            for metric in metrics:
                value = comparison_results[method].get(metric, 0)
                if metric in ['psnr', 'snr']:
                    report += f"{value:<10.2f}"
                else:
                    report += f"{value:<10.4f}"
            report += "\n"
        
        report += "\n"
        
        # Detailed analysis
        report += "Detailed Analysis:\n"
        report += "-" * 20 + "\n"
        
        for method in methods:
            metrics = comparison_results[method]
            report += f"\n{method}:\n"
            report += f"  PSNR: {metrics['psnr']:.2f} dB\n"
            report += f"  SSIM: {metrics['ssim']:.4f}\n"
            report += f"  MSE: {metrics['mse']:.6f}\n"
            report += f"  SNR: {metrics['snr']:.2f} dB\n"
            report += f"  Correlation: {metrics['correlation']:.4f}\n"
            report += f"  Contrast Improvement: {metrics['contrast_improvement']:.4f}\n"
        
        # Save report if file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {output_file}")
        
        return report
    
    def plot_comparison(self, 
                       original: np.ndarray,
                       processed_images: Dict[str, np.ndarray],
                       save_path: Optional[str] = None):
        """
        Create visualization of comparison results.
        
        Args:
            original: Original image
            processed_images: Dictionary of processed images
            save_path: Optional path to save plot
        """
        n_methods = len(processed_images)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
        
        # Original image
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Processed images
        for i, (method_name, processed_image) in enumerate(processed_images.items()):
            axes[0, i + 1].imshow(processed_image, cmap='gray')
            axes[0, i + 1].set_title(method_name)
            axes[0, i + 1].axis('off')
            
            # Difference image
            diff = np.abs(original - processed_image)
            axes[1, i + 1].imshow(diff, cmap='hot')
            axes[1, i + 1].set_title(f'{method_name} - Diff')
            axes[1, i + 1].axis('off')
        
        # Original difference (should be zero)
        axes[1, 0].imshow(np.zeros_like(original), cmap='hot')
        axes[1, 0].set_title('Original - Diff')
        axes[1, 0].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        if image.max() > 1.0:
            return image / 255.0
        return image.copy()


def evaluate_image_quality(original: np.ndarray,
                          processed: np.ndarray) -> Dict:
    """
    Convenience function for image quality evaluation.
    
    Args:
        original: Original image
        processed: Processed image
        
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = ImageEvaluator()
    return evaluator.evaluate_all_metrics(original, processed)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create test images
    original = np.random.rand(100, 100).astype(np.float32)
    
    # Create processed images with different quality levels
    processed1 = original + 0.1 * np.random.randn(100, 100)  # Low noise
    processed2 = original + 0.3 * np.random.randn(100, 100)  # High noise
    
    # Evaluate quality
    evaluator = ImageEvaluator()
    
    metrics1 = evaluator.evaluate_all_metrics(original, processed1)
    metrics2 = evaluator.evaluate_all_metrics(original, processed2)
    
    print("Low noise image metrics:")
    print(f"PSNR: {metrics1['psnr']:.2f} dB")
    print(f"SSIM: {metrics1['ssim']:.4f}")
    
    print("\nHigh noise image metrics:")
    print(f"PSNR: {metrics2['psnr']:.2f} dB")
    print(f"SSIM: {metrics2['ssim']:.4f}")
    
    # Compare methods
    processed_images = {
        'Low Noise': processed1,
        'High Noise': processed2
    }
    
    comparison = evaluator.compare_methods(original, processed_images)
    report = evaluator.generate_evaluation_report(comparison)
    print("\n" + report) 