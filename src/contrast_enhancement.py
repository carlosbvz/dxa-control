"""
Contrast Enhancement Module for DXA Image Processing
===================================================

This module implements various contrast enhancement techniques for DXA images,
including histogram equalization, adaptive histogram equalization, and
contrast stretching methods.

Author: Carlos Benavides
Course: SP-2141 Teoría de la detección y estimación
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Union
import logging
from scipy import ndimage
from skimage import exposure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContrastEnhancer:
    """
    Contrast enhancement for DXA images.
    
    This class provides various methods for improving image contrast,
    which is crucial for better visualization and analysis of DXA images.
    """
    
    def __init__(self):
        """Initialize the contrast enhancer."""
        logger.info("Contrast enhancer initialized")
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to enhance contrast.
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            Enhanced image
        """
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            image_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image_norm = image.copy()
        
        # Apply histogram equalization
        enhanced = cv2.equalizeHist(image_norm)
        
        logger.info("Histogram equalization applied")
        return enhanced
    
    def adaptive_histogram_equalization(self, 
                                      image: np.ndarray,
                                      clip_limit: float = 2.0,
                                      tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply adaptive histogram equalization (CLAHE).
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            Enhanced image
        """
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            image_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image_norm = image.copy()
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Apply CLAHE
        enhanced = clahe.apply(image_norm)
        
        logger.info(f"Adaptive histogram equalization applied with clip_limit={clip_limit}")
        return enhanced
    
    def contrast_stretching(self, 
                           image: np.ndarray,
                           lower_percentile: float = 2.0,
                           upper_percentile: float = 98.0) -> np.ndarray:
        """
        Apply contrast stretching using percentile-based normalization.
        
        Args:
            image: Input image
            lower_percentile: Lower percentile for stretching
            upper_percentile: Upper percentile for stretching
            
        Returns:
            Enhanced image
        """
        # Calculate percentiles
        p_low = np.percentile(image, lower_percentile)
        p_high = np.percentile(image, upper_percentile)
        
        # Apply contrast stretching
        enhanced = np.clip((image - p_low) / (p_high - p_low), 0, 1)
        
        logger.info(f"Contrast stretching applied: {lower_percentile}%-{upper_percentile}%")
        return enhanced
    
    def gamma_correction(self, 
                        image: np.ndarray,
                        gamma: float = 1.2) -> np.ndarray:
        """
        Apply gamma correction for contrast enhancement.
        
        Args:
            image: Input image (normalized to [0,1])
            gamma: Gamma value (gamma > 1 darkens, gamma < 1 brightens)
            
        Returns:
            Enhanced image
        """
        # Ensure image is in [0,1] range
        if image.max() > 1.0:
            image = image / 255.0
        
        # Apply gamma correction
        enhanced = np.power(image, gamma)
        
        logger.info(f"Gamma correction applied with gamma={gamma}")
        return enhanced
    
    def unsharp_masking(self, 
                       image: np.ndarray,
                       sigma: float = 1.0,
                       amount: float = 1.0,
                       threshold: float = 0.0) -> np.ndarray:
        """
        Apply unsharp masking for edge enhancement.
        
        Args:
            image: Input image
            sigma: Standard deviation for Gaussian blur
            amount: Amount of sharpening
            threshold: Threshold for sharpening
            
        Returns:
            Enhanced image
        """
        # Apply Gaussian blur
        blurred = ndimage.gaussian_filter(image, sigma=sigma)
        
        # Calculate sharpened image
        sharpened = image + amount * (image - blurred)
        
        # Apply threshold
        if threshold > 0:
            sharpened = np.where(np.abs(image - blurred) > threshold, sharpened, image)
        
        # Ensure values are in valid range
        enhanced = np.clip(sharpened, 0, 1)
        
        logger.info(f"Unsharp masking applied with sigma={sigma}, amount={amount}")
        return enhanced
    
    def multi_scale_enhancement(self, 
                               image: np.ndarray,
                               scales: list = [1, 2, 4]) -> np.ndarray:
        """
        Apply multi-scale contrast enhancement.
        
        Args:
            image: Input image
            scales: List of scales for enhancement
            
        Returns:
            Enhanced image
        """
        enhanced = np.zeros_like(image)
        
        for scale in scales:
            # Create Gaussian kernel
            kernel_size = 2 * scale + 1
            kernel = cv2.getGaussianKernel(kernel_size, scale)
            kernel_2d = kernel * kernel.T
            
            # Apply Gaussian blur
            blurred = cv2.filter2D(image, -1, kernel_2d)
            
            # Calculate difference
            diff = image - blurred
            
            # Add to enhanced image
            enhanced += diff / len(scales)
        
        # Normalize result
        enhanced = np.clip(enhanced, 0, 1)
        
        logger.info(f"Multi-scale enhancement applied with scales={scales}")
        return enhanced
    
    def adaptive_contrast_enhancement(self, 
                                    image: np.ndarray,
                                    window_size: int = 15) -> np.ndarray:
        """
        Apply adaptive contrast enhancement based on local statistics.
        
        Args:
            image: Input image
            window_size: Size of local window
            
        Returns:
            Enhanced image
        """
        # Calculate local mean and standard deviation
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        local_mean = cv2.filter2D(image, -1, kernel)
        local_var = cv2.filter2D(image**2, -1, kernel) - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 1e-6))
        
        # Calculate global statistics
        global_mean = np.mean(image)
        global_std = np.std(image)
        
        # Adaptive enhancement
        enhanced = local_mean + (image - local_mean) * (global_std / local_std)
        
        # Ensure values are in valid range
        enhanced = np.clip(enhanced, 0, 1)
        
        logger.info(f"Adaptive contrast enhancement applied with window_size={window_size}")
        return enhanced


def enhance_contrast(image: np.ndarray, 
                    method: str = 'clahe',
                    **kwargs) -> np.ndarray:
    """
    Convenience function for contrast enhancement.
    
    Args:
        image: Input image
        method: Enhancement method
        **kwargs: Additional arguments for the method
        
    Returns:
        Enhanced image
    """
    enhancer = ContrastEnhancer()
    
    methods = {
        'histogram': enhancer.histogram_equalization,
        'clahe': enhancer.adaptive_histogram_equalization,
        'stretching': enhancer.contrast_stretching,
        'gamma': enhancer.gamma_correction,
        'unsharp': enhancer.unsharp_masking,
        'multiscale': enhancer.multi_scale_enhancement,
        'adaptive': enhancer.adaptive_contrast_enhancement
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available methods: {list(methods.keys())}")
    
    return methods[method](image, **kwargs)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create a test image with low contrast
    test_image = np.random.rand(100, 100).astype(np.float32)
    test_image = test_image * 0.3 + 0.2  # Low contrast image
    
    # Apply different enhancement methods
    enhancer = ContrastEnhancer()
    
    enhanced_clahe = enhancer.adaptive_histogram_equalization(test_image)
    enhanced_stretch = enhancer.contrast_stretching(test_image)
    enhanced_gamma = enhancer.gamma_correction(test_image)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(test_image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(enhanced_clahe, cmap='gray')
    axes[0, 1].set_title('CLAHE')
    axes[1, 0].imshow(enhanced_stretch, cmap='gray')
    axes[1, 0].set_title('Contrast Stretching')
    axes[1, 1].imshow(enhanced_gamma, cmap='gray')
    axes[1, 1].set_title('Gamma Correction')
    
    plt.tight_layout()
    plt.show() 