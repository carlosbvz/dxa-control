
# Adaptive Control Optimization Report
====================================

## Summary Statistics
- Total iterations: 50
- Best PSNR: 23.47 dB
- Best SSIM: 0.9452
- Average PSNR: 16.78 dB
- Average SSIM: 0.8174
- Total processing time: 11.78 seconds

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
- Min: 13.44 dB
- Max: 23.47 dB
- Std: 2.88 dB

### SSIM Statistics:
- Min: 0.7103
- Max: 0.9452
- Std: 0.0721

## Top 5 Results

### Rank 1
- PSNR: 23.47 dB
- SSIM: 0.9452
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 1}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (4, 4)}

### Rank 2
- PSNR: 23.47 dB
- SSIM: 0.9452
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 5}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (4, 4)}

### Rank 3
- PSNR: 23.47 dB
- SSIM: 0.9452
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 10}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (4, 4)}

### Rank 4
- PSNR: 23.47 dB
- SSIM: 0.9452
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 20}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (4, 4)}

### Rank 5
- PSNR: 21.50 dB
- SSIM: 0.9183
- Kalman: {'process_noise': 5, 'measurement_noise': 0.001, 'initial_uncertainty': 1}
- Contrast: {'clip_limit': 1.0, 'tile_grid_size': (8, 8)}
