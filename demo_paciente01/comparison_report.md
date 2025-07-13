
# Pipeline Comparison Report
===========================

## Processing Times
- Classic Pipeline: 2.85 seconds
- Adaptive Pipeline: 125.51 seconds
- Time Difference: 122.65 seconds

## Quality Metrics Comparison

### Classic Pipeline:

- PSNR: 28.70 dB
- SSIM: 0.5886
- MSE: 0.001350

### Adaptive Pipeline:

- PSNR: 35.62 dB
- SSIM: 0.7928
- MSE: 0.000274

## Optimization Results:
- Best PSNR Found: 35.62 dB
- Best SSIM Found: 0.7928
- Optimization Time: 124.95 seconds
- Best Kalman Parameters: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 1}
- Best Contrast Parameters: {'clip_limit': 1.0, 'tile_grid_size': (4, 4)}

## Improvements:
- PSNR Improvement: +6.93 dB
- SSIM Improvement: +0.2042
- Relative PSNR Improvement: +24.1%
- Relative SSIM Improvement: +34.7%

## File Structure:
```
demo_paciente01/
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
