"""
BMD (Bone Mineral Density) Extraction Module
===========================================

This module implements bone mineral density extraction and analysis
from DXA images using detected regions of interest.

Author: Carlos Benavides
Course: SP-2141 Teoría de la detección y estimación
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from sklearn.cluster import KMeans
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BMDExtractor:
    """
    Bone Mineral Density (BMD) extraction and analysis.
    
    This class provides methods for extracting BMD values from DXA images
    and performing statistical analysis on the extracted data.
    """
    
    def __init__(self):
        """Initialize the BMD extractor."""
        logger.info("BMD extractor initialized")
    
    def extract_bmd_values(self, 
                          image: np.ndarray,
                          mask: np.ndarray,
                          method: str = 'direct') -> Dict:
        """
        Extract BMD values from image using ROI mask.
        
        Args:
            image: Input DXA image
            mask: ROI mask (bone regions)
            method: Extraction method ('direct', 'histogram', 'clustering')
            
        Returns:
            Dictionary containing BMD statistics
        """
        methods = {
            'direct': self._extract_direct,
            'histogram': self._extract_histogram,
            'clustering': self._extract_clustering,
            'percentile': self._extract_percentile
        }
        
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Available methods: {list(methods.keys())}")
        
        return methods[method](image, mask)
    
    def _extract_direct(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Extract BMD values directly from masked regions.
        
        Args:
            image: Input image
            mask: ROI mask
            
        Returns:
            Dictionary of BMD statistics
        """
        # Extract pixel values from bone regions
        bone_pixels = image[mask]
        
        if len(bone_pixels) == 0:
            logger.warning("No bone pixels found in mask")
            return self._empty_bmd_stats()
        
        # Calculate basic statistics
        stats_dict = {
            'mean_bmd': np.mean(bone_pixels),
            'std_bmd': np.std(bone_pixels),
            'median_bmd': np.median(bone_pixels),
            'min_bmd': np.min(bone_pixels),
            'max_bmd': np.max(bone_pixels),
            'total_pixels': len(bone_pixels),
            'bone_area': np.sum(mask),
            'pixel_values': bone_pixels.tolist()
        }
        
        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90, 95]
        for p in percentiles:
            stats_dict[f'bmd_p{p}'] = np.percentile(bone_pixels, p)
        
        logger.info(f"Direct BMD extraction completed: mean={stats_dict['mean_bmd']:.4f}")
        return stats_dict
    
    def _extract_histogram(self, 
                          image: np.ndarray, 
                          mask: np.ndarray,
                          bins: int = 50) -> Dict:
        """
        Extract BMD values using histogram analysis.
        
        Args:
            image: Input image
            mask: ROI mask
            bins: Number of histogram bins
            
        Returns:
            Dictionary of BMD statistics
        """
        # Extract bone pixels
        bone_pixels = image[mask]
        
        if len(bone_pixels) == 0:
            return self._empty_bmd_stats()
        
        # Calculate histogram
        hist, bin_edges = np.histogram(bone_pixels, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find peak of histogram (most common BMD value)
        peak_idx = np.argmax(hist)
        peak_bmd = bin_centers[peak_idx]
        
        # Calculate statistics
        stats_dict = {
            'peak_bmd': peak_bmd,
            'histogram_peaks': bin_centers[hist > np.max(hist) * 0.8].tolist(),
            'histogram_bins': bin_centers.tolist(),
            'histogram_values': hist.tolist(),
            'mean_bmd': np.mean(bone_pixels),
            'std_bmd': np.std(bone_pixels),
            'total_pixels': len(bone_pixels)
        }
        
        logger.info(f"Histogram BMD extraction completed: peak={peak_bmd:.4f}")
        return stats_dict
    
    def _extract_clustering(self, 
                           image: np.ndarray,
                           mask: np.ndarray,
                           n_clusters: int = 3) -> Dict:
        """
        Extract BMD values using clustering analysis.
        
        Args:
            image: Input image
            mask: ROI mask
            n_clusters: Number of clusters
            
        Returns:
            Dictionary of BMD statistics
        """
        # Extract bone pixels
        bone_pixels = image[mask]
        
        if len(bone_pixels) == 0:
            return self._empty_bmd_stats()
        
        # Reshape for clustering
        pixels_2d = bone_pixels.reshape(-1, 1)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(pixels_2d)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_pixels = bone_pixels[labels == i]
            cluster_stats[f'cluster_{i}'] = {
                'mean': np.mean(cluster_pixels),
                'std': np.std(cluster_pixels),
                'size': len(cluster_pixels),
                'percentage': len(cluster_pixels) / len(bone_pixels) * 100
            }
        
        # Overall statistics
        stats_dict = {
            'mean_bmd': np.mean(bone_pixels),
            'std_bmd': np.std(bone_pixels),
            'total_pixels': len(bone_pixels),
            'clusters': cluster_stats,
            'cluster_centers': kmeans.cluster_centers_.flatten().tolist()
        }
        
        logger.info(f"Clustering BMD extraction completed with {n_clusters} clusters")
        return stats_dict
    
    def _extract_percentile(self, 
                           image: np.ndarray,
                           mask: np.ndarray,
                           percentiles: List[float] = None) -> Dict:
        """
        Extract BMD values using percentile analysis.
        
        Args:
            image: Input image
            mask: ROI mask
            percentiles: List of percentiles to calculate
            
        Returns:
            Dictionary of BMD statistics
        """
        if percentiles is None:
            percentiles = [5, 10, 25, 50, 75, 90, 95]
        
        # Extract bone pixels
        bone_pixels = image[mask]
        
        if len(bone_pixels) == 0:
            return self._empty_bmd_stats()
        
        # Calculate percentiles
        percentile_values = {}
        for p in percentiles:
            percentile_values[f'p{p}'] = np.percentile(bone_pixels, p)
        
        # Calculate additional statistics
        stats_dict = {
            'mean_bmd': np.mean(bone_pixels),
            'std_bmd': np.std(bone_pixels),
            'median_bmd': np.median(bone_pixels),
            'total_pixels': len(bone_pixels),
            'percentiles': percentile_values,
            'iqr': percentile_values['p75'] - percentile_values['p25']
        }
        
        logger.info(f"Percentile BMD extraction completed")
        return stats_dict
    
    def _empty_bmd_stats(self) -> Dict:
        """Return empty BMD statistics."""
        return {
            'mean_bmd': 0.0,
            'std_bmd': 0.0,
            'median_bmd': 0.0,
            'min_bmd': 0.0,
            'max_bmd': 0.0,
            'total_pixels': 0,
            'bone_area': 0
        }
    
    def analyze_bmd_distribution(self, 
                               bmd_values: List[float]) -> Dict:
        """
        Analyze the distribution of BMD values.
        
        Args:
            bmd_values: List of BMD values
            
        Returns:
            Dictionary of distribution analysis
        """
        if not bmd_values:
            return {'error': 'No BMD values provided'}
        
        bmd_array = np.array(bmd_values)
        
        # Basic statistics
        analysis = {
            'mean': np.mean(bmd_array),
            'std': np.std(bmd_array),
            'median': np.median(bmd_array),
            'skewness': stats.skew(bmd_array),
            'kurtosis': stats.kurtosis(bmd_array),
            'range': np.max(bmd_array) - np.min(bmd_array),
            'iqr': np.percentile(bmd_array, 75) - np.percentile(bmd_array, 25)
        }
        
        # Test for normality
        try:
            _, p_value = stats.normaltest(bmd_array)
            analysis['normality_test_p'] = p_value
            analysis['is_normal'] = p_value > 0.05
        except:
            analysis['normality_test_p'] = None
            analysis['is_normal'] = None
        
        logger.info(f"BMD distribution analysis completed: mean={analysis['mean']:.4f}")
        return analysis
    
    def compare_bmd_regions(self, 
                           image: np.ndarray,
                           masks: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Compare BMD values across different regions.
        
        Args:
            image: Input image
            masks: Dictionary of region masks
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for region_name, mask in masks.items():
            bmd_stats = self.extract_bmd_values(image, mask, method='direct')
            
            results.append({
                'region': region_name,
                'mean_bmd': bmd_stats['mean_bmd'],
                'std_bmd': bmd_stats['std_bmd'],
                'median_bmd': bmd_stats['median_bmd'],
                'total_pixels': bmd_stats['total_pixels'],
                'bone_area': bmd_stats['bone_area']
            })
        
        df = pd.DataFrame(results)
        
        # Add statistical comparisons
        if len(results) > 1:
            bmd_values = [result['mean_bmd'] for result in results]
            df['z_score'] = [(bmd - np.mean(bmd_values)) / np.std(bmd_values) for bmd in bmd_values]
        
        logger.info(f"BMD comparison completed for {len(masks)} regions")
        return df
    
    def calculate_bmd_zscore(self, 
                           patient_bmd: float,
                           reference_mean: float,
                           reference_std: float) -> float:
        """
        Calculate Z-score for BMD comparison.
        
        Args:
            patient_bmd: Patient's BMD value
            reference_mean: Reference population mean
            reference_std: Reference population standard deviation
            
        Returns:
            Z-score
        """
        z_score = (patient_bmd - reference_mean) / reference_std
        logger.info(f"BMD Z-score calculated: {z_score:.3f}")
        return z_score
    
    def calculate_t_score(self, 
                         patient_bmd: float,
                         young_adult_mean: float,
                         young_adult_std: float) -> float:
        """
        Calculate T-score for BMD comparison.
        
        Args:
            patient_bmd: Patient's BMD value
            young_adult_mean: Young adult reference mean
            young_adult_std: Young adult reference standard deviation
            
        Returns:
            T-score
        """
        t_score = (patient_bmd - young_adult_mean) / young_adult_std
        logger.info(f"BMD T-score calculated: {t_score:.3f}")
        return t_score
    
    def classify_bmd_status(self, t_score: float) -> str:
        """
        Classify BMD status based on T-score.
        
        Args:
            t_score: T-score value
            
        Returns:
            Classification string
        """
        if t_score >= -1.0:
            classification = "Normal"
        elif t_score >= -2.5:
            classification = "Osteopenia"
        else:
            classification = "Osteoporosis"
        
        logger.info(f"BMD status classified as: {classification}")
        return classification


def extract_bmd(image: np.ndarray, 
                mask: np.ndarray,
                method: str = 'direct',
                **kwargs) -> Dict:
    """
    Convenience function for BMD extraction.
    
    Args:
        image: Input image
        mask: ROI mask
        method: Extraction method
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of BMD statistics
    """
    extractor = BMDExtractor()
    return extractor.extract_bmd_values(image, mask, method, **kwargs)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create a test image with different bone density regions
    test_image = np.random.rand(100, 100).astype(np.float32)
    
    # Add regions with different densities
    test_image[20:40, 30:70] = 0.8  # High density
    test_image[60:80, 20:60] = 0.6  # Medium density
    
    # Create masks for different regions
    mask1 = np.zeros_like(test_image, dtype=bool)
    mask1[20:40, 30:70] = True
    
    mask2 = np.zeros_like(test_image, dtype=bool)
    mask2[60:80, 20:60] = True
    
    # Extract BMD values
    extractor = BMDExtractor()
    
    bmd1 = extractor.extract_bmd_values(test_image, mask1)
    bmd2 = extractor.extract_bmd_values(test_image, mask2)
    
    print("Region 1 BMD:", bmd1['mean_bmd'])
    print("Region 2 BMD:", bmd2['mean_bmd'])
    
    # Compare regions
    masks = {'region1': mask1, 'region2': mask2}
    comparison = extractor.compare_bmd_regions(test_image, masks)
    print("\nBMD Comparison:")
    print(comparison) 