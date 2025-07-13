
# Pipeline Comparison Report
===========================

## Processing Times
- Classic Pipeline: 3.11 seconds
- Adaptive Pipeline: 125.92 seconds
- Time Difference: 122.81 seconds

## Quality Metrics Comparison

### Classic Pipeline:

- PSNR: 19.73 dB
- SSIM: 0.6240
- MSE: 0.010640

### Adaptive Pipeline:

- PSNR: 25.15 dB
- SSIM: 0.7952
- MSE: 0.003053

## Optimization Results:
- Best PSNR Found: 25.15 dB
- Best SSIM Found: 0.7952
- Optimization Time: 125.45 seconds
- Best Kalman Parameters: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 1}
- Best Contrast Parameters: {'clip_limit': 1.0, 'tile_grid_size': (16, 16)}

## Improvements:
- PSNR Improvement: +5.42 dB
- SSIM Improvement: +0.1712
- Relative PSNR Improvement: +27.5%
- Relative SSIM Improvement: +27.4%

## File Structure:
```
demo_noise_HE/
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
