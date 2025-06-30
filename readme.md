code/
├── data/
│   ├── raw/          # Imágenes HE/LE originales
│   └── masks/        # Máscaras manuales
├── results/          # Salidas (imágenes procesadas, CSV, gráficos)
├── src/
│   ├── kalman_filter.py
│   ├── contrast_enhancement.py
│   ├── roi_tracking.py
│   ├── bmd_extraction.py
│   ├── evaluation.py
│   ├── utils.py
│   └── main.py
├── tests/            # Pruebas unitarias (test_*.py)
├── notebooks/        # (Opcional) Notebooks para prototipos y visualización
├── README.md         # Documentación de instalación y uso
└── requirements.txt  # Dependencias (`pip freeze > requirements.txt`)