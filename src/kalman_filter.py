"""
Kalman Filter Module for DXA Image Processing
=============================================

This module implements a spatial Kalman filter for noise reduction and contrast
enhancement in DXA images. The filter estimates the true pixel values from
noisy observations using Bayesian estimation principles.

Author: Carlos Benavides
Course: SP-2141 Teoría de la detección y estimación
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalmanFilterImage:
    """
    Spatial Kalman Filter for image processing.
    
    This class implements a Kalman filter that processes images pixel by pixel,
    estimating the true pixel values, noise variance, and contrast factors.
    """
    
    def __init__(self, 
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1,
                 initial_uncertainty: float = 0.1):
        """
        Initialize the Kalman filter.
        
        Args:
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
            initial_uncertainty: Initial state uncertainty
        """
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance
        self.P0 = initial_uncertainty  # Initial state uncertainty
        
        # State vector: [pixel_value, noise_variance, contrast_factor]
        self.state_dim = 3
        
        # Initialize state transition matrix (identity for spatial filtering)
        self.F = np.eye(self.state_dim)
        
        # Initialize observation matrix (we only observe pixel values)
        self.H = np.array([[1, 0, 0]])
        
        logger.info(f"Kalman filter initialized with Q={self.Q}, R={self.R}")
    
    def filter_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Kalman filter to the entire image.
        
        Args:
            image: Input image (grayscale, float32, normalized to [0,1])
            
        Returns:
            Filtered image
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Normalize image to [0,1] if not already
        if image.max() > 1.0:
            image = image / 255.0
        
        height, width = image.shape
        logger.info(f"Processing image of size {height}x{width}")
        
        # Initialize state and covariance matrices
        x = np.zeros((self.state_dim, height, width))
        P = np.eye(self.state_dim) * self.P0
        
        # Initialize state with image values
        x[0, :, :] = image.copy()
        x[1, :, :] = 0.1  # Initial noise variance estimate
        x[2, :, :] = 1.0  # Initial contrast factor
        
        # Process image pixel by pixel
        for i in range(height):
            for j in range(width):
                # Prediction step
                x_pred = self.F @ x[:, i, j]
                P_pred = self.F @ P @ self.F.T + self.Q * np.eye(self.state_dim)
                
                # Update step
                z = image[i, j]  # Current observation
                S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
                K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain
                
                # Update state and covariance
                x[:, i, j] = x_pred + K @ (z - self.H @ x_pred)
                P = (np.eye(self.state_dim) - K @ self.H) @ P_pred
        
        # Return filtered pixel values
        filtered_image = x[0, :, :]
        
        # Ensure values are in valid range
        filtered_image = np.clip(filtered_image, 0, 1)
        
        logger.info("Kalman filtering completed")
        return filtered_image
    
    def filter_image_adaptive(self, image: np.ndarray, 
                            window_size: int = 5) -> np.ndarray:
        """
        Apply adaptive Kalman filter using local window statistics.
        
        Args:
            image: Input image
            window_size: Size of local window for statistics
            
        Returns:
            Filtered image
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        if image.max() > 1.0:
            image = image / 255.0
        
        height, width = image.shape
        filtered_image = np.zeros_like(image)
        
        # Calculate local statistics
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        local_mean = cv2.filter2D(image, -1, kernel)
        local_var = cv2.filter2D(image**2, -1, kernel) - local_mean**2
        
        # Apply adaptive filtering
        for i in range(height):
            for j in range(width):
                # Adaptive noise estimation
                local_noise = max(local_var[i, j], 0.001)
                
                # Update measurement noise based on local statistics
                adaptive_R = local_noise
                
                # Apply single-pixel Kalman filter
                filtered_image[i, j] = self._filter_pixel(
                    image[i, j], local_mean[i, j], adaptive_R
                )
        
        logger.info("Adaptive Kalman filtering completed")
        return filtered_image
    
    def _filter_pixel(self, observation: float, 
                     prior_estimate: float, 
                     measurement_noise: float) -> float:
        """
        Apply Kalman filter to a single pixel.
        
        Args:
            observation: Current pixel observation
            prior_estimate: Prior estimate of pixel value
            measurement_noise: Measurement noise for this pixel
            
        Returns:
            Filtered pixel value
        """
        # Prediction
        x_pred = prior_estimate
        P_pred = self.P0 + self.Q
        
        # Update
        K = P_pred / (P_pred + measurement_noise)  # Kalman gain
        x_updated = x_pred + K * (observation - x_pred)
        
        return x_updated


def apply_kalman_filter(image: np.ndarray, 
                       method: str = 'standard',
                       **kwargs) -> np.ndarray:
    """
    Convenience function to apply Kalman filtering to an image.
    
    Args:
        image: Input image
        method: Filtering method ('standard' or 'adaptive')
        **kwargs: Additional arguments for the filter
        
    Returns:
        Filtered image
    """
    kf = KalmanFilterImage(**kwargs)
    
    if method == 'adaptive':
        return kf.filter_image_adaptive(image)
    else:
        return kf.filter_image(image)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create a test image with noise
    test_image = np.random.rand(100, 100).astype(np.float32)
    noisy_image = test_image + 0.1 * np.random.randn(100, 100)
    noisy_image = np.clip(noisy_image, 0, 1)
    
    # Apply Kalman filter
    kf = KalmanFilterImage()
    filtered_image = kf.filter_image(noisy_image)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(noisy_image, cmap='gray')
    axes[1].set_title('Noisy')
    axes[2].imshow(filtered_image, cmap='gray')
    axes[2].set_title('Filtered')
    
    plt.tight_layout()
    plt.show() 