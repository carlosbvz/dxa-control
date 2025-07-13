
# Adaptive Control Optimization Report
====================================

## Summary Statistics
- Total iterations: 50
- Best PSNR: 35.62 dB
- Best SSIM: 0.7928
- Average PSNR: 27.69 dB
- Average SSIM: 0.5258
- Total processing time: 122.26 seconds

## Best Parameters Found
### Kalman Filter Parameters:
- Process Noise (Q): 5
- Measurement Noise (R): 0.001
- Initial Uncertainty (P0): 1

### Contrast Enhancement Parameters:
- CLAHE Clip Limit: 1.0
- Tile Grid Size: (4, 4)

## Parameter Analysis
### PSNR Statistics:
- Min: 21.85 dB
- Max: 35.62 dB
- Std: 4.03 dB

### SSIM Statistics:
- Min: 0.3035
- Max: 0.7928
- Std: 0.1694

## Top 5 Results

### Rank 1
- PSNR: 35.62 dB
- SSIM: 0.7928
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 1}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (4, 4)}

### Rank 2
- PSNR: 35.62 dB
- SSIM: 0.7928
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 5}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (4, 4)}

### Rank 3
- PSNR: 35.62 dB
- SSIM: 0.7928
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 10}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (4, 4)}

### Rank 4
- PSNR: 35.62 dB
- SSIM: 0.7928
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 20}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (4, 4)}

### Rank 5
- PSNR: 33.01 dB
- SSIM: 0.7782
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 1}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (8, 8)}
