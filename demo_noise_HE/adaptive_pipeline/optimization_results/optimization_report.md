
# Adaptive Control Optimization Report
====================================

## Summary Statistics
- Total iterations: 50
- Best PSNR: 25.15 dB
- Best SSIM: 0.7952
- Average PSNR: 19.53 dB
- Average SSIM: 0.5719
- Total processing time: 123.19 seconds

## Best Parameters Found
### Kalman Filter Parameters:
- Process Noise (Q): 5
- Measurement Noise (R): 0.001
- Initial Uncertainty (P0): 1

### Contrast Enhancement Parameters:
- CLAHE Clip Limit: 1.0
- Tile Grid Size: (16, 16)

## Parameter Analysis
### PSNR Statistics:
- Min: 15.77 dB
- Max: 25.15 dB
- Std: 3.25 dB

### SSIM Statistics:
- Min: 0.3474
- Max: 0.8361
- Std: 0.1615

## Top 5 Results

### Rank 1
- PSNR: 25.15 dB
- SSIM: 0.7952
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 1}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (16, 16)}

### Rank 2
- PSNR: 25.15 dB
- SSIM: 0.7952
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 5}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (16, 16)}

### Rank 3
- PSNR: 25.15 dB
- SSIM: 0.7952
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 10}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (16, 16)}

### Rank 4
- PSNR: 25.15 dB
- SSIM: 0.7952
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 20}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (16, 16)}

### Rank 5
- PSNR: 24.69 dB
- SSIM: 0.8110
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 1}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (8, 8)}
