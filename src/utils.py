"""
Utilities Module for DXA Image Processing
========================================

This module provides utility functions for image loading, saving,
preprocessing, and other common operations used throughout the project.

Author: Carlos Benavides
Course: SP-2141 Teoría de la detección y estimación
"""

import numpy as np
import cv2
import os
import glob
from typing import List, Tuple, Optional, Union
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageUtils:
    """
    Utility functions for image processing operations.
    """
    
    @staticmethod
    def load_image(file_path: str, 
                   normalize: bool = True,
                   target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            file_path: Path to image file
            normalize: Whether to normalize to [0, 1] range
            target_size: Optional target size for resizing
            
        Returns:
            Loaded image as numpy array
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Load image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        
        # Resize if specified
        if target_size is not None:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize if requested
        if normalize:
            image = image.astype(np.float32) / 255.0
        
        logger.info(f"Image loaded: {file_path}, shape: {image.shape}")
        return image
    
    @staticmethod
    def save_image(image: np.ndarray, 
                   file_path: str,
                   normalize: bool = True) -> None:
        """
        Save image to file.
        
        Args:
            image: Image to save
            file_path: Output file path
            normalize: Whether to normalize to [0, 255] range
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Prepare image for saving
        if normalize and image.max() <= 1.0:
            save_image = (image * 255).astype(np.uint8)
        else:
            save_image = image.astype(np.uint8)
        
        # Save image
        success = cv2.imwrite(file_path, save_image)
        
        if not success:
            raise ValueError(f"Could not save image: {file_path}")
        
        logger.info(f"Image saved: {file_path}")
    
    @staticmethod
    def load_image_batch(directory: str, 
                        pattern: str = "*.png",
                        normalize: bool = True,
                        target_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Load multiple images from directory.
        
        Args:
            directory: Directory containing images
            pattern: File pattern to match
            normalize: Whether to normalize images
            target_size: Optional target size for resizing
            
        Returns:
            List of loaded images
        """
        # Find image files
        search_pattern = os.path.join(directory, pattern)
        image_files = glob.glob(search_pattern)
        
        if not image_files:
            raise FileNotFoundError(f"No images found matching pattern: {search_pattern}")
        
        # Load images
        images = []
        for file_path in sorted(image_files):
            try:
                image = ImageUtils.load_image(file_path, normalize, target_size)
                images.append(image)
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
        
        logger.info(f"Loaded {len(images)} images from {directory}")
        return images
    
    @staticmethod
    def preprocess_image(image: np.ndarray,
                        normalize: bool = True,
                        denoise: bool = False,
                        denoise_strength: float = 10.0) -> np.ndarray:
        """
        Preprocess image for analysis.
        
        Args:
            image: Input image
            normalize: Whether to normalize
            denoise: Whether to apply denoising
            denoise_strength: Denoising strength
            
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        # Normalize
        if normalize and processed.max() > 1.0:
            processed = processed / 255.0
        
        # Denoise if requested
        if denoise:
            processed = cv2.fastNlMeansDenoising(
                (processed * 255).astype(np.uint8),
                None,
                denoise_strength,
                7, 21
            ).astype(np.float32) / 255.0
        
        logger.info("Image preprocessing completed")
        return processed
    
    @staticmethod
    def create_noisy_image(image: np.ndarray,
                          noise_type: str = 'gaussian',
                          noise_level: float = 0.1) -> np.ndarray:
        """
        Create noisy version of image for testing.
        
        Args:
            image: Original image
            noise_type: Type of noise ('gaussian', 'salt_pepper', 'poisson')
            noise_level: Noise level/intensity
            
        Returns:
            Noisy image
        """
        noisy = image.copy()
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, image.shape)
            noisy = noisy + noise
        
        elif noise_type == 'salt_pepper':
            # Salt noise
            salt_mask = np.random.random(image.shape) < noise_level / 2
            noisy[salt_mask] = 1.0
            
            # Pepper noise
            pepper_mask = np.random.random(image.shape) < noise_level / 2
            noisy[pepper_mask] = 0.0
        
        elif noise_type == 'poisson':
            noise = np.random.poisson(noise_level * 255, image.shape) / 255.0
            noisy = noisy + noise
        
        # Ensure values are in valid range
        noisy = np.clip(noisy, 0, 1)
        
        logger.info(f"Created noisy image with {noise_type} noise, level={noise_level}")
        return noisy
    
    @staticmethod
    def calculate_image_statistics(image: np.ndarray) -> dict:
        """
        Calculate basic image statistics.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'mean': np.mean(image),
            'std': np.std(image),
            'min': np.min(image),
            'max': np.max(image),
            'median': np.median(image),
            'shape': image.shape,
            'dtype': str(image.dtype)
        }
        
        # Calculate percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for p in percentiles:
            stats[f'p{p}'] = np.percentile(image, p)
        
        logger.info(f"Image statistics calculated: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        return stats
    
    @staticmethod
    def visualize_images(images: List[np.ndarray],
                        titles: Optional[List[str]] = None,
                        figsize: Tuple[int, int] = (15, 5),
                        save_path: Optional[str] = None) -> None:
        """
        Visualize multiple images side by side.
        
        Args:
            images: List of images to display
            titles: Optional list of titles
            figsize: Figure size
            save_path: Optional path to save figure
        """
        n_images = len(images)
        
        if titles is None:
            titles = [f'Image {i+1}' for i in range(n_images)]
        
        fig, axes = plt.subplots(1, n_images, figsize=figsize)
        
        if n_images == 1:
            axes = [axes]
        
        for i, (image, title) in enumerate(zip(images, titles)):
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_histogram(image: np.ndarray,
                      bins: int = 50,
                      title: str = "Image Histogram",
                      save_path: Optional[str] = None) -> None:
        """
        Plot image histogram.
        
        Args:
            image: Input image
            bins: Number of histogram bins
            title: Plot title
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 6))
        plt.hist(image.flatten(), bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Histogram saved to {save_path}")
        
        plt.show()


class DataUtils:
    """
    Utility functions for data management and file operations.
    """
    
    @staticmethod
    def save_results(results: dict, 
                    file_path: str,
                    format: str = 'json') -> None:
        """
        Save results to file.
        
        Args:
            results: Results dictionary
            file_path: Output file path
            format: File format ('json' or 'pickle')
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                elif isinstance(value, np.integer):
                    json_results[key] = int(value)
                elif isinstance(value, np.floating):
                    json_results[key] = float(value)
                else:
                    json_results[key] = value
            
            with open(file_path, 'w') as f:
                json.dump(json_results, f, indent=2)
        
        elif format == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(results, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results saved to {file_path}")
    
    @staticmethod
    def load_results(file_path: str,
                    format: str = 'json') -> dict:
        """
        Load results from file.
        
        Args:
            file_path: Input file path
            format: File format ('json' or 'pickle')
            
        Returns:
            Loaded results dictionary
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Results file not found: {file_path}")
        
        if format == 'json':
            with open(file_path, 'r') as f:
                results = json.load(f)
        
        elif format == 'pickle':
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results loaded from {file_path}")
        return results
    
    @staticmethod
    def create_experiment_directory(base_path: str,
                                   experiment_name: str) -> str:
        """
        Create directory structure for experiment.
        
        Args:
            base_path: Base directory path
            experiment_name: Name of experiment
            
        Returns:
            Path to experiment directory
        """
        experiment_path = os.path.join(base_path, experiment_name)
        
        # Create subdirectories
        subdirs = ['data', 'results', 'plots', 'logs']
        for subdir in subdirs:
            os.makedirs(os.path.join(experiment_path, subdir), exist_ok=True)
        
        logger.info(f"Experiment directory created: {experiment_path}")
        return experiment_path
    
    @staticmethod
    def get_file_list(directory: str,
                     pattern: str = "*",
                     recursive: bool = False) -> List[str]:
        """
        Get list of files matching pattern.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        search_pattern = os.path.join(directory, pattern)
        
        if recursive:
            files = glob.glob(search_pattern, recursive=True)
        else:
            files = glob.glob(search_pattern)
        
        # Sort files for consistent ordering
        files.sort()
        
        logger.info(f"Found {len(files)} files matching pattern: {pattern}")
        return files


class ValidationUtils:
    """
    Utility functions for input validation and error checking.
    """
    
    @staticmethod
    def validate_image(image: np.ndarray,
                      expected_shape: Optional[Tuple] = None,
                      expected_dtype: Optional[type] = None) -> bool:
        """
        Validate image properties.
        
        Args:
            image: Image to validate
            expected_shape: Expected image shape
            expected_dtype: Expected data type
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        if image.ndim != 2:
            raise ValueError("Image must be 2-dimensional")
        
        if expected_shape is not None and image.shape != expected_shape:
            raise ValueError(f"Image shape {image.shape} does not match expected {expected_shape}")
        
        if expected_dtype is not None and image.dtype != expected_dtype:
            raise ValueError(f"Image dtype {image.dtype} does not match expected {expected_dtype}")
        
        return True
    
    @staticmethod
    def validate_file_path(file_path: str,
                          must_exist: bool = True) -> bool:
        """
        Validate file path.
        
        Args:
            file_path: File path to validate
            must_exist: Whether file must exist
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if not isinstance(file_path, str):
            raise ValueError("File path must be a string")
        
        if must_exist and not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        return True
    
    @staticmethod
    def validate_directory(directory: str,
                          must_exist: bool = True,
                          create_if_missing: bool = False) -> bool:
        """
        Validate directory path.
        
        Args:
            directory: Directory path to validate
            must_exist: Whether directory must exist
            create_if_missing: Whether to create directory if missing
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if not isinstance(directory, str):
            raise ValueError("Directory path must be a string")
        
        if not os.path.exists(directory):
            if create_if_missing:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            elif must_exist:
                raise FileNotFoundError(f"Directory does not exist: {directory}")
        
        return True


# Convenience functions
def load_dxa_image(file_path: str) -> np.ndarray:
    """Load and preprocess DXA image."""
    return ImageUtils.load_image(file_path, normalize=True)


def save_processed_image(image: np.ndarray, file_path: str) -> None:
    """Save processed image."""
    ImageUtils.save_image(image, file_path, normalize=True)


def create_test_image(size: Tuple[int, int] = (100, 100),
                     noise_level: float = 0.1) -> np.ndarray:
    """Create test image with noise."""
    # Create base image
    image = np.random.rand(*size).astype(np.float32)
    
    # Add some structure
    x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1, size[0]))
    image += 0.3 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    
    # Add noise
    if noise_level > 0:
        image = ImageUtils.create_noisy_image(image, 'gaussian', noise_level)
    
    return image


if __name__ == "__main__":
    # Example usage
    # Create test image
    test_image = create_test_image((100, 100), noise_level=0.1)
    
    # Calculate statistics
    stats = ImageUtils.calculate_image_statistics(test_image)
    print("Image Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Visualize image and histogram
    ImageUtils.visualize_images([test_image], titles=['Test Image'])
    ImageUtils.plot_histogram(test_image, title="Test Image Histogram") 