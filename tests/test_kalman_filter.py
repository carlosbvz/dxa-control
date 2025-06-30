"""

Test module for Kalman filter functionality.

"""



import unittest

import numpy as np

import sys

import os

import time



# Add src directory to path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))



# Import all modules here so they are available for all tests

from kalman_filter import KalmanFilterImage, apply_kalman_filter

from contrast_enhancement import ContrastEnhancer, enhance_contrast

from roi_tracking import ROITracker, detect_roi

from bmd_extraction import BMDExtractor, extract_bmd

from evaluation import ImageEvaluator, evaluate_image_quality

from utils import ImageUtils, DataUtils, ValidationUtils

from main import DXAPipeline





class TestKalmanFilter(unittest.TestCase):

    """Test cases for Kalman filter functionality."""

    

    def setUp(self):

        """Set up test fixtures."""

        self.test_image = np.random.rand(50, 50).astype(np.float32)

        self.kf = KalmanFilterImage()

    

    def test_kalman_filter_initialization(self):

        """Test Kalman filter initialization."""

        self.assertIsNotNone(self.kf)

        self.assertEqual(self.kf.state_dim, 3)

        self.assertEqual(self.kf.F.shape, (3, 3))

        self.assertEqual(self.kf.H.shape, (1, 3))

    

    def test_filter_image_output_shape(self):

        """Test that filtered image has same shape as input."""

        filtered = self.kf.filter_image(self.test_image)

        self.assertEqual(filtered.shape, self.test_image.shape)

    

    def test_filter_image_value_range(self):

        """Test that filtered image values are in valid range."""

        filtered = self.kf.filter_image(self.test_image)

        self.assertTrue(np.all(filtered >= 0))

        self.assertTrue(np.all(filtered <= 1))

    

    def test_adaptive_filter(self):

        """Test adaptive Kalman filtering."""

        filtered = self.kf.filter_image_adaptive(self.test_image, window_size=5)

        self.assertEqual(filtered.shape, self.test_image.shape)

        self.assertTrue(np.all(filtered >= 0))

        self.assertTrue(np.all(filtered <= 1))

    

    def test_convenience_function(self):

        """Test convenience function."""

        filtered = apply_kalman_filter(self.test_image, method='standard')

        self.assertEqual(filtered.shape, self.test_image.shape)

    

    def test_noise_reduction(self):

        """Test that filtering reduces noise."""

        # Create noisy image

        noisy_image = self.test_image + 0.2 * np.random.randn(*self.test_image.shape)

        noisy_image = np.clip(noisy_image, 0, 1)

        

        # Apply filter

        filtered = self.kf.filter_image(noisy_image)

        

        # Check that filtered image is closer to original than noisy

        original_diff = np.mean(np.abs(self.test_image - noisy_image))

        filtered_diff = np.mean(np.abs(self.test_image - filtered))

        

        # Filtered should be closer to original (lower difference)

        self.assertLess(filtered_diff, original_diff)





if __name__ == '__main__':

    unittest.main() 
