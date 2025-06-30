#!/usr/bin/env python3
"""
Test script for DXA Image Processing Pipeline
=============================================

This script tests the complete pipeline to ensure all components work correctly.

Author: Carlos Benavides
Course: SP-2141 Teoría de la detección y estimación
"""

import sys
import os
import numpy as np
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from kalman_filter import KalmanFilterImage, apply_kalman_filter
        print("✓ Kalman filter module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import kalman_filter: {e}")
        return False
    
    try:
        from contrast_enhancement import ContrastEnhancer, enhance_contrast
        print("✓ Contrast enhancement module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import contrast_enhancement: {e}")
        return False
    
    try:
        from roi_tracking import ROITracker, detect_roi
        print("✓ ROI tracking module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import roi_tracking: {e}")
        return False
    
    try:
        from bmd_extraction import BMDExtractor, extract_bmd
        print("✓ BMD extraction module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import bmd_extraction: {e}")
        return False
    
    try:
        from evaluation import ImageEvaluator, evaluate_image_quality
        print("✓ Evaluation module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluation: {e}")
        return False
    
    try:
        from utils import ImageUtils, DataUtils, ValidationUtils
        print("✓ Utils module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    try:
        from main import DXAPipeline
        print("✓ Main pipeline module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import main: {e}")
        return False
    
    return True

def test_individual_components():
    """Test individual components."""
    print("\nTesting individual components...")
    
    # Create test image
    test_image = np.random.rand(100, 100).astype(np.float32)
    test_image = test_image * 0.5 + 0.2  # Normalize to reasonable range
    
    # Test Kalman filter
    try:
        kf = KalmanFilterImage()
        filtered = kf.filter_image(test_image)
        print("✓ Kalman filter works")
    except Exception as e:
        print(f"✗ Kalman filter failed: {e}")
        return False
    
    # Test contrast enhancement
    try:
        enhancer = ContrastEnhancer()
        enhanced = enhancer.adaptive_histogram_equalization(filtered)
        print("✓ Contrast enhancement works")
    except Exception as e:
        print(f"✗ Contrast enhancement failed: {e}")
        return False
    
    # Test ROI detection
    try:
        tracker = ROITracker()
        roi_mask = tracker.detect_bone_regions(enhanced, method='otsu')
        print("✓ ROI detection works")
    except Exception as e:
        print(f"✗ ROI detection failed: {e}")
        return False
    
    # Test BMD extraction
    try:
        extractor = BMDExtractor()
        bmd_stats = extractor.extract_bmd_values(enhanced, roi_mask)
        print("✓ BMD extraction works")
    except Exception as e:
        print(f"✗ BMD extraction failed: {e}")
        return False
    
    # Test evaluation
    try:
        evaluator = ImageEvaluator()
        metrics = evaluator.evaluate_all_metrics(test_image, enhanced)
        print("✓ Evaluation works")
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return False
    
    return True

def test_complete_pipeline():
    """Test the complete pipeline."""
    print("\nTesting complete pipeline...")
    
    try:
        # Create test image
        test_image = ImageUtils.create_test_image((150, 150), noise_level=0.15)
        
        # Save test image
        test_input = "test_input.png"
        ImageUtils.save_image(test_image, test_input)
        
        # Initialize and run pipeline
        pipeline = DXAPipeline()
        results = pipeline.process_single_image(test_input, "test_output")
        
        # Check results
        if 'processing_time' in results and 'bmd' in results:
            print("✓ Complete pipeline works")
            print(f"  Processing time: {results['processing_time']:.2f} seconds")
            print(f"  Mean BMD: {results['bmd']['statistics']['mean_bmd']:.4f}")
            
            # Clean up
            os.remove(test_input)
            import shutil
            shutil.rmtree("test_output", ignore_errors=True)
            
            return True
        else:
            print("✗ Pipeline results incomplete")
            return False
            
    except Exception as e:
        print(f"✗ Complete pipeline failed: {e}")
        return False

def main():
    """Run all tests."""
    print("DXA Image Processing Pipeline - Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        return False
    
    # Test individual components
    if not test_individual_components():
        print("\n❌ Component tests failed.")
        return False
    
    # Test complete pipeline
    if not test_complete_pipeline():
        print("\n❌ Pipeline test failed.")
        return False
    
    print("\n✅ All tests passed! The pipeline is working correctly.")
    print("\nYou can now use the pipeline with your DXA images:")
    print("  python src/main.py input_image.png output_directory")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 