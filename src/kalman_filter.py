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
    
    CONCEPTO CLAVE DEL CURSO: Este filtro implementa estimación recursiva bayesiana
    aplicando el teorema de Bayes para actualizar la estimación del estado (valor del píxel)
    con cada nueva observación (píxel ruidoso).
    """
    def __init__(self,
                 process_noise: float = 1e-5,
                 measurement_noise: float = 0.2,
                 initial_uncertainty: float = 1.0,
                 passes: int = 2):
        """
        Inicialización del filtro de Kalman con parámetros clave.
        
        CONCEPTOS DEL CURSO APLICADOS:
        - process_noise (Q): Incertidumbre del modelo de estado
        - measurement_noise (R): Incertidumbre de la observación
        - initial_uncertainty (P0): Incertidumbre inicial a priori
        
        Args:
            process_noise:       Q (process variance, on [0,1] scale) - Ruido del proceso
            measurement_noise:   R (measurement variance, on [0,1] scale) - Ruido de medición
            initial_uncertainty: initial P (variance) at start of each scanline - Incertidumbre inicial
            passes:              how many horizontal+vertical sweeps to do - Número de pasadas
        """
        # PARÁMETROS DEL FILTRO DE KALMAN (CONCEPTOS FUNDAMENTALES DEL CURSO)
        self.Q = process_noise        # Q: Varianza del ruido del proceso (modelo de estado)
        self.R = measurement_noise    # R: Varianza del ruido de medición (observaciones)
        self.P0 = initial_uncertainty # P0: Incertidumbre inicial (varianza a priori)
        self.passes = max(1, passes)  # Número de pasadas horizontales + verticales
        
        logger.info(f"KalmanFilterImage(Q={self.Q}, R={self.R}, P0={self.P0}, passes={self.passes})")

    def filter_image(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica el filtro de Kalman en múltiples pasadas: horizontal y luego vertical.
        
        ALGORITMO DE ESTIMACIÓN RECURSIVA:
        1. Predicción: x̂(k|k-1) = x̂(k-1|k-1) (estado anterior)
        2. Actualización: x̂(k|k) = x̂(k|k-1) + K(k)[z(k) - x̂(k|k-1)]
        3. Ganancia de Kalman: K(k) = P(k|k-1) / [P(k|k-1) + R]
        
        Args:
            image: grayscale, uint8 or float32 - Imagen de entrada

        Returns:
            filtered image (float32, [0,1]) - Imagen filtrada
        """
        # PREPROCESAMIENTO: Normalización de la imagen a rango [0,1]
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

        H, W = img.shape  # Dimensiones de la imagen
        result = img.copy()  # Copia para procesamiento iterativo

        # PASADAS MÚLTIPLES: Cada pasada mejora la estimación
        for sweep in range(self.passes):
            # PASADA HORIZONTAL: Procesa fila por fila (estimación recursiva)
            horiz = np.zeros_like(result)
            for i in range(H):  # Para cada fila
                # INICIALIZACIÓN DEL ESTADO: Primer píxel como estado inicial
                x = result[i, 0]       # initial state = first pixel
                P = self.P0            # Incertidumbre inicial
                
                # ESTIMACIÓN RECURSIVA PIXEL POR PIXEL (CONCEPTO CLAVE)
                for j in range(W):  # Para cada píxel en la fila
                    # PASO 1: PREDICCIÓN (Prediction Step)
                    # x̂(k|k-1) = F·x̂(k-1|k-1) donde F=1 (modelo de estado simple)
                    # P(k|k-1) = F·P(k-1|k-1)·F^T + Q
                    P_pred = P + self.Q  # Predicción de la incertidumbre

                    # PASO 2: OBSERVACIÓN (Measurement)
                    z = result[i, j]  # Observación actual (píxel ruidoso)

                    # PASO 3: ACTUALIZACIÓN DE KALMAN (Update Step)
                    # Ganancia de Kalman: K(k) = P(k|k-1) / [P(k|k-1) + R]
                    K = P_pred / (P_pred + self.R)
                    
                    # Estimación actualizada: x̂(k|k) = x̂(k|k-1) + K(k)[z(k) - x̂(k|k-1)]
                    x = x + K * (z - x)
                    
                    # Incertidumbre actualizada: P(k|k) = (1 - K(k))·P(k|k-1)
                    P = (1 - K) * P_pred

                    # Guarda el píxel estimado
                    horiz[i, j] = x

            # PASADA VERTICAL: Procesa columna por columna (segunda dimensión)
            vert = np.zeros_like(horiz)
            for j in range(W):  # Para cada columna
                # Reinicialización para la pasada vertical
                x = horiz[0, j]  # Estado inicial = primer píxel de la columna
                P = self.P0      # Incertidumbre inicial
                
                # ESTIMACIÓN RECURSIVA VERTICAL
                for i in range(H):  # Para cada píxel en la columna
                    # Mismo proceso de predicción-actualización
                    P_pred = P + self.Q  # Predicción
                    z = horiz[i, j]      # Observación

                    # Actualización de Kalman
                    K = P_pred / (P_pred + self.R)
                    x = x + K * (z - x)
                    P = (1 - K) * P_pred

                    vert[i, j] = x

            # ACTUALIZACIÓN DEL RESULTADO: Cada pasada mejora la estimación
            result = vert

        # POSTPROCESAMIENTO: Asegura valores en rango válido
        return np.clip(result, 0.0, 1.0)


def apply_kalman_filter(image: np.ndarray,
                        process_noise: float = 1e-5,
                        measurement_noise: float = 0.2,
                        initial_uncertainty: float = 1.0,
                        passes: int = 2) -> np.ndarray:
    """
    Punto de entrada conveniente para aplicar el filtro de Kalman.
    
    INTERFAZ SIMPLIFICADA: Permite usar el filtro sin crear instancia explícita.
    
    Args:
        image: Imagen de entrada
        process_noise: Q - Ruido del proceso (pequeño para estabilidad)
        measurement_noise: R - Ruido de medición (controla suavizado)
        initial_uncertainty: P0 - Incertidumbre inicial
        passes: Número de pasadas horizontales + verticales
    
    Returns:
        Imagen filtrada con ruido reducido
    """
    # CREACIÓN Y APLICACIÓN DEL FILTRO
    kf = KalmanFilterImage(process_noise=process_noise,
                           measurement_noise=measurement_noise,
                           initial_uncertainty=initial_uncertainty,
                           passes=passes)
    return kf.filter_image(image)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # DEMOSTRACIÓN DEL FILTRO DE KALMAN
    # 1) Carga y normalización de imagen de prueba
    clean = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # 2) Adición de ruido fuerte (σ = 0.2) para simular condiciones DXA
    noise_std = 0.2
    noisy = np.clip(clean + noise_std * np.random.randn(*clean.shape), 0, 1)

    # 3) Aplicación del filtro de Kalman con parámetros optimizados
    # R = 0.2 (suavizado moderado) y 2 pasadas
    filtered = apply_kalman_filter(noisy,
                                   process_noise=1e-5,    # Q pequeño para estabilidad
                                   measurement_noise=0.2, # R controla nivel de suavizado
                                   passes=2)              # 2 pasadas para mejor resultado

    # 4) Visualización lado a lado para comparación
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes,
                              [clean, noisy, filtered],
                              ['Original', 'Noisy (σ=0.2)', 'Kalman Smoothed']):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
