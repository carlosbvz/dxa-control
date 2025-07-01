"""
Kalman Filter Module for DXA Image Processing (working)
=======================================================

Spatial Kalman filter implemented as repeated 1-D recursive passes
for noise reduction in single-channel images.

Author: Carlos Benavides (revised)
Course: SP-2141 Teoría de la detección y estimación
"""

import numpy as np
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KalmanFilterImage:
    """
    Spatial Kalman Filter for image processing.

    Applies multiple 1-D Kalman sweeps along rows and then columns
    to achieve 2-D smoothing.
    """
    def __init__(self,
                 process_noise: float = 1e-5,
                 measurement_noise: float = 0.2,
                 initial_uncertainty: float = 1.0,
                 passes: int = 2):
        """
        Args:
            process_noise:       Q (process variance, on [0,1] scale)
            measurement_noise:   R (measurement variance, on [0,1] scale)
            initial_uncertainty: initial P (variance) at start of each scanline
            passes:              how many horizontal+vertical sweeps to do
        """
        self.Q = process_noise
        self.R = measurement_noise
        self.P0 = initial_uncertainty
        self.passes = max(1, passes)
        logger.info(f"KalmanFilterImage(Q={self.Q}, R={self.R}, P0={self.P0}, passes={self.passes})")

    def filter_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the filter in 'passes' sweeps: horizontal then vertical.

        Args:
            image: grayscale, uint8 or float32

        Returns:
            filtered image (float32, [0,1])
        """
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

        H, W = img.shape
        result = img.copy()

        for sweep in range(self.passes):
            # Horizontal pass
            horiz = np.zeros_like(result)
            for i in range(H):
                x = result[i, 0]       # initial state = first pixel
                P = self.P0
                for j in range(W):
                    # predict
                    P_pred = P + self.Q

                    # observe
                    z = result[i, j]

                    # Kalman update
                    K = P_pred / (P_pred + self.R)
                    x = x + K * (z - x)
                    P = (1 - K) * P_pred

                    horiz[i, j] = x

            # Vertical pass
            vert = np.zeros_like(horiz)
            for j in range(W):
                x = horiz[0, j]
                P = self.P0
                for i in range(H):
                    P_pred = P + self.Q
                    z = horiz[i, j]

                    K = P_pred / (P_pred + self.R)
                    x = x + K * (z - x)
                    P = (1 - K) * P_pred

                    vert[i, j] = x

            result = vert

        return np.clip(result, 0.0, 1.0)


def apply_kalman_filter(image: np.ndarray,
                        process_noise: float = 1e-5,
                        measurement_noise: float = 0.2,
                        initial_uncertainty: float = 1.0,
                        passes: int = 2) -> np.ndarray:
    """
    Convenience entry point.
    """
    kf = KalmanFilterImage(process_noise=process_noise,
                           measurement_noise=measurement_noise,
                           initial_uncertainty=initial_uncertainty,
                           passes=passes)
    return kf.filter_image(image)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1) Load and normalize a test image
    clean = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # 2) Add strong noise (σ = 0.2)
    noise_std = 0.2
    noisy = np.clip(clean + noise_std * np.random.randn(*clean.shape), 0, 1)

    # 3) Apply Kalman smoothing with R = 0.2 (moderate smoothing) and 2 passes
    filtered = apply_kalman_filter(noisy,
                                   process_noise=1e-5,
                                   measurement_noise=0.2,
                                   passes=2)

    # 4) Display side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes,
                              [clean, noisy, filtered],
                              ['Original', 'Noisy (σ=0.2)', 'Kalman Smoothed']):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
