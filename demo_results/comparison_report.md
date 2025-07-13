
# Pipeline Comparison Report
===========================

## Processing Times
- Classic Pipeline: 0.42 seconds
- Adaptive Pipeline: 12.30 seconds
- Time Difference: 11.88 seconds

## Quality Metrics Comparison

### Classic Pipeline:

- PSNR: 16.03 dB
- SSIM: 0.7932
- MSE: 0.024964

### Adaptive Pipeline:

- PSNR: 23.47 dB
- SSIM: 0.9452
- MSE: 0.004495

## Optimization Results:
- Best PSNR Found: 23.47 dB
- Best SSIM Found: 0.9452
- Optimization Time: 12.26 seconds
- Best Kalman Parameters: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 1}
- Best Contrast Parameters: {'clip_limit': 1.0, 'tile_grid_size': (4, 4)}

## Improvements:
- PSNR Improvement: +7.45 dB
- SSIM Improvement: +0.1520
- Relative PSNR Improvement: +46.5%
- Relative SSIM Improvement: +19.2%

## File Structure:
```
demo_results/
├── test_image.png
├── classic_pipeline/
│   ├── kalman_filtered.png
│   ├── contrast_enhanced.png
│   ├── roi_mask.png
│   └── summary.txt
└── adaptive_pipeline/
    ├── optimization_results/
    │   ├── iteration_001/
    │   ├── iteration_002/
    │   ├── ...
    │   ├── optimization_report.md
    │   └── detailed_results.json
    ├── kalman_filtered.png
    ├── contrast_enhanced.png
    ├── roi_mask.png
    └── summary.txt
```

## Conclusion:
The adaptive control system demonstrates modern control principles by:
1. Automatically adapting parameters to each image's characteristics
2. Optimizing quality metrics (PSNR, SSIM) through systematic search
3. Providing comprehensive documentation of the optimization process
4. Eliminating manual parameter tuning while improving results

This approach aligns with SP-2159 course objectives by implementing:
- Adaptive control systems
- Parameter optimization
- Experimental validation
- Automated decision making
