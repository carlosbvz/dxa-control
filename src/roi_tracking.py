"""
ROI Tracking Module for DXA Image Processing
===========================================

This module implements region of interest (ROI) detection and tracking
for DXA images, focusing on bone region detection and segmentation.

Author: Carlos Benavides
Course: SP-2141 Teoría de la detección y estimación
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict
import logging
from scipy import ndimage
from skimage import measure, morphology
from skimage.filters import threshold_otsu, threshold_adaptive

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ROITracker:
    """
    Region of Interest (ROI) detection and tracking for DXA images.
    
    This class provides methods for detecting bone regions and tracking
    them across different processing stages.
    """
    
    def __init__(self):
        """Initialize the ROI tracker."""
        logger.info("ROI tracker initialized")
    
    def detect_bone_regions(self, 
                           image: np.ndarray,
                           method: str = 'otsu',
                           **kwargs) -> np.ndarray:
        """
        Detect bone regions in DXA images.
        
        Args:
            image: Input image (grayscale)
            method: Detection method ('otsu', 'adaptive', 'kmeans', 'watershed')
            **kwargs: Additional arguments for the method
            
        Returns:
            Binary mask of bone regions
        """
        methods = {
            'otsu': self._detect_otsu,
            'adaptive': self._detect_adaptive,
            'kmeans': self._detect_kmeans,
            'watershed': self._detect_watershed,
            'morphological': self._detect_morphological
        }
        
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Available methods: {list(methods.keys())}")
        
        return methods[method](image, **kwargs)
    
    def _detect_otsu(self, image: np.ndarray) -> np.ndarray:
        """
        Detect bone regions using Otsu's thresholding.
        
        Args:
            image: Input image
            
        Returns:
            Binary mask
        """
        # Normalize image to 0-255 range
        if image.max() <= 1.0:
            image_norm = (image * 255).astype(np.uint8)
        else:
            image_norm = image.astype(np.uint8)
        
        # Apply Otsu thresholding
        threshold = threshold_otsu(image_norm)
        mask = image_norm > threshold
        
        # Apply morphological operations
        mask = self._post_process_mask(mask)
        
        logger.info(f"Otsu detection completed with threshold={threshold}")
        return mask
    
    def _detect_adaptive(self, 
                        image: np.ndarray,
                        block_size: int = 15,
                        c: int = 2) -> np.ndarray:
        """
        Detect bone regions using adaptive thresholding.
        
        Args:
            image: Input image
            block_size: Size of local neighborhood
            c: Constant subtracted from mean
            
        Returns:
            Binary mask
        """
        # Normalize image to 0-255 range
        if image.max() <= 1.0:
            image_norm = (image * 255).astype(np.uint8)
        else:
            image_norm = image.astype(np.uint8)
        
        # Apply adaptive thresholding
        mask = threshold_adaptive(image_norm, block_size, offset=c)
        
        # Apply morphological operations
        mask = self._post_process_mask(mask)
        
        logger.info(f"Adaptive detection completed with block_size={block_size}, c={c}")
        return mask
    
    def _detect_kmeans(self, 
                      image: np.ndarray,
                      n_clusters: int = 3) -> np.ndarray:
        """
        Detect bone regions using K-means clustering.
        
        Args:
            image: Input image
            n_clusters: Number of clusters
            
        Returns:
            Binary mask
        """
        from sklearn.cluster import KMeans
        
        # Reshape image for clustering
        pixels = image.reshape(-1, 1)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(pixels)
        
        # Find the cluster with highest mean intensity (bone)
        cluster_means = [np.mean(pixels[labels == i]) for i in range(n_clusters)]
        bone_cluster = np.argmax(cluster_means)
        
        # Create mask
        mask = (labels == bone_cluster).reshape(image.shape)
        
        # Apply morphological operations
        mask = self._post_process_mask(mask)
        
        logger.info(f"K-means detection completed with {n_clusters} clusters")
        return mask
    
    def _detect_watershed(self, 
                         image: np.ndarray,
                         markers: int = 10) -> np.ndarray:
        """
        Detect bone regions using watershed segmentation.
        
        Args:
            image: Input image
            markers: Number of markers for watershed
            
        Returns:
            Binary mask
        """
        # Normalize image to 0-255 range
        if image.max() <= 1.0:
            image_norm = (image * 255).astype(np.uint8)
        else:
            image_norm = image.astype(np.uint8)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image_norm, (5, 5), 0)
        
        # Create markers
        markers_img = np.zeros_like(image_norm)
        markers_img[image_norm > np.percentile(image_norm, 90)] = 255
        markers_img[image_norm < np.percentile(image_norm, 10)] = 128
        
        # Apply watershed
        markers = cv2.connectedComponents(markers_img.astype(np.uint8))[1]
        watershed_result = cv2.watershed(blurred, markers)
        
        # Create mask (watershed result == 255 indicates watershed lines)
        mask = watershed_result != 255
        mask = mask.astype(bool)
        
        # Apply morphological operations
        mask = self._post_process_mask(mask)
        
        logger.info("Watershed detection completed")
        return mask
    
    def _detect_morphological(self, 
                            image: np.ndarray,
                            kernel_size: int = 5) -> np.ndarray:
        """
        Detect bone regions using morphological operations.
        
        Args:
            image: Input image
            kernel_size: Size of morphological kernel
            
        Returns:
            Binary mask
        """
        # Normalize image to 0-255 range
        if image.max() <= 1.0:
            image_norm = (image * 255).astype(np.uint8)
        else:
            image_norm = image.astype(np.uint8)
        
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply morphological operations
        # Top-hat transform to enhance bright regions
        tophat = cv2.morphologyEx(image_norm, cv2.MORPH_TOPHAT, kernel)
        
        # Threshold the top-hat result
        threshold = np.mean(tophat) + 2 * np.std(tophat)
        mask = tophat > threshold
        
        # Apply morphological operations
        mask = self._post_process_mask(mask)
        
        logger.info(f"Morphological detection completed with kernel_size={kernel_size}")
        return mask
    
    def _post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Post-process binary mask with morphological operations.
        
        Args:
            mask: Input binary mask
            
        Returns:
            Processed mask
        """
        # Remove small objects
        mask = morphology.remove_small_objects(mask, min_size=100)
        
        # Fill holes
        mask = morphology.remove_small_holes(mask, area_threshold=50)
        
        # Apply closing operation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return mask.astype(bool)
    
    def track_roi(self, 
                 image_sequence: List[np.ndarray],
                 initial_mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Track ROI across a sequence of images.
        
        Args:
            image_sequence: List of images
            initial_mask: Initial ROI mask (optional)
            
        Returns:
            List of tracked masks
        """
        masks = []
        
        for i, image in enumerate(image_sequence):
            if i == 0 and initial_mask is not None:
                # Use provided initial mask
                mask = initial_mask.copy()
            else:
                # Detect ROI in current image
                mask = self.detect_bone_regions(image, method='otsu')
            
            # Apply tracking constraints if not first image
            if i > 0 and len(masks) > 0:
                mask = self._apply_tracking_constraints(mask, masks[-1])
            
            masks.append(mask)
            logger.info(f"ROI tracked in frame {i+1}/{len(image_sequence)}")
        
        return masks
    
    def _apply_tracking_constraints(self, 
                                  current_mask: np.ndarray,
                                  previous_mask: np.ndarray) -> np.ndarray:
        """
        Apply tracking constraints to maintain consistency.
        
        Args:
            current_mask: Current frame mask
            previous_mask: Previous frame mask
            
        Returns:
            Constrained mask
        """
        # Calculate overlap
        overlap = np.logical_and(current_mask, previous_mask)
        overlap_ratio = np.sum(overlap) / np.sum(previous_mask)
        
        # If overlap is too low, use previous mask as constraint
        if overlap_ratio < 0.3:
            # Use morphological dilation of previous mask as constraint
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_previous = cv2.dilate(previous_mask.astype(np.uint8), kernel)
            
            # Combine with current detection
            constrained_mask = np.logical_and(current_mask, dilated_previous.astype(bool))
            
            logger.info(f"Applied tracking constraint, overlap_ratio={overlap_ratio:.3f}")
            return constrained_mask
        
        return current_mask
    
    def extract_roi_properties(self, 
                             image: np.ndarray,
                             mask: np.ndarray) -> Dict:
        """
        Extract properties of detected ROI.
        
        Args:
            image: Input image
            mask: ROI mask
            
        Returns:
            Dictionary of ROI properties
        """
        # Calculate basic properties
        roi_pixels = image[mask]
        
        properties = {
            'area': np.sum(mask),
            'mean_intensity': np.mean(roi_pixels),
            'std_intensity': np.std(roi_pixels),
            'min_intensity': np.min(roi_pixels),
            'max_intensity': np.max(roi_pixels),
            'centroid': ndimage.center_of_mass(mask),
            'bbox': self._get_bounding_box(mask)
        }
        
        # Calculate shape properties
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)
        
        if regions:
            region = regions[0]  # Largest region
            properties.update({
                'eccentricity': region.eccentricity,
                'solidity': region.solidity,
                'extent': region.extent,
                'perimeter': region.perimeter
            })
        
        logger.info(f"ROI properties extracted: area={properties['area']}")
        return properties
    
    def _get_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box of ROI.
        
        Args:
            mask: ROI mask
            
        Returns:
            Bounding box (x, y, width, height)
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return (0, 0, 0, 0)
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return (cmin, rmin, cmax - cmin + 1, rmax - rmin + 1)


def detect_roi(image: np.ndarray, 
              method: str = 'otsu',
              **kwargs) -> np.ndarray:
    """
    Convenience function for ROI detection.
    
    Args:
        image: Input image
        method: Detection method
        **kwargs: Additional arguments for the method
        
    Returns:
        Binary mask of detected ROI
    """
    tracker = ROITracker()
    return tracker.detect_bone_regions(image, method, **kwargs)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create a test image with bone-like regions
    test_image = np.random.rand(100, 100).astype(np.float32)
    
    # Add some bright regions to simulate bone
    test_image[20:40, 30:70] = 0.8
    test_image[60:80, 20:60] = 0.9
    
    # Apply different detection methods
    tracker = ROITracker()
    
    mask_otsu = tracker.detect_bone_regions(test_image, method='otsu')
    mask_adaptive = tracker.detect_bone_regions(test_image, method='adaptive')
    mask_kmeans = tracker.detect_bone_regions(test_image, method='kmeans')
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(test_image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(mask_otsu, cmap='gray')
    axes[0, 1].set_title('Otsu Detection')
    axes[1, 0].imshow(mask_adaptive, cmap='gray')
    axes[1, 0].set_title('Adaptive Detection')
    axes[1, 1].imshow(mask_kmeans, cmap='gray')
    axes[1, 1].set_title('K-means Detection')
    
    plt.tight_layout()
    plt.show() 